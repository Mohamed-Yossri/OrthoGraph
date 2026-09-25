import json
import sqlite3
import threading
from pathlib import Path
from domain import now


class Store:
    def __init__(self, data_dir: Path):
        self.root = data_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(data_dir/'cases.sqlite3'), check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL, stage TEXT NOT NULL, error TEXT, report TEXT,
            options TEXT NOT NULL)''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(cases)')}
        if 'owner_id' not in columns:
            self.db.execute('ALTER TABLE cases ADD COLUMN owner_id INTEGER NOT NULL DEFAULT 1')
        # A process interruption is visible rather than an indefinitely spinning job.
        self.db.execute("UPDATE cases SET status='error', error='The server restarted during processing. Please analyze the image again.' WHERE status IN ('processing','queued','finalizing')")
        self.db.commit()

    def create(self, case_id, name, options, owner_id=1):
        with self.lock:
            self.db.execute('INSERT INTO cases (id,name,created_at,status,stage,error,report,options,owner_id) VALUES (?,?,?,?,?,?,?,?,?)',
                (case_id, name, now(), 'queued', 'queued', None, None, json.dumps(options),owner_id))
            self.db.commit()

    def get(self, case_id, owner_id=None):
        with self.lock:
            self.db.row_factory = sqlite3.Row
            row = self.db.execute('SELECT * FROM cases WHERE id=?' + (' AND owner_id=?' if owner_id is not None else ''),
                                  (case_id, owner_id) if owner_id is not None else (case_id,)).fetchone()
            if not row:
                raise KeyError(case_id)
            result = dict(row)
            result['report'] = json.loads(result['report']) if result['report'] else None
            result['options'] = json.loads(result['options'])
            return result

    def list(self, owner_id=None):
        with self.lock:
            self.db.row_factory = sqlite3.Row
            rows = self.db.execute('SELECT id,name,created_at,status,stage,error FROM cases' +
                (' WHERE owner_id=?' if owner_id is not None else '') + ' ORDER BY created_at DESC LIMIT 100',
                (owner_id,) if owner_id is not None else ()).fetchall()
            return [dict(r) for r in rows]

    def update(self, case_id, **values):
        if set(values) - {'status','stage','error','report'}:
            raise ValueError('Invalid case column')
        if 'report' in values:
            values['report'] = json.dumps(values['report'])
        with self.lock:
            self.db.execute('UPDATE cases SET '+','.join(k+'=?' for k in values)+' WHERE id=?',
                            [*values.values(),case_id])
            self.db.commit()

    def directory(self, case_id):
        # IDs are never interpreted as a path from an unvalidated request.
        from uuid import UUID
        safe = str(UUID(case_id))
        return self.root/'cases'/safe

    def close(self):
        self.db.close()

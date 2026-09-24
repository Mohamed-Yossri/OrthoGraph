"""Small local account store with Argon2 passwords and revocable sessions."""
from __future__ import annotations
import hashlib
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

OWNER_USERNAME = 'mohamed-yossri'
COOKIE = 'orthograph_session'
SESSION_SECONDS = 7 * 24 * 3600
MAX_ACCOUNTS = 25
USERNAME_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{2,31}$')


class AuthStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / 'auth.sqlite3'
        self.bootstrap_path = self.dir / 'initial-login.txt'
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS accounts (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL, must_change INTEGER NOT NULL DEFAULT 0)')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, expires INTEGER NOT NULL)')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(sessions)')}
        if 'account_id' not in columns:
            self.db.execute('ALTER TABLE sessions ADD COLUMN account_id INTEGER NOT NULL DEFAULT 1')
        self.db.commit()
        self.db_path.chmod(0o600)
        self.lock = threading.RLock()
        self.hasher = PasswordHasher()
        self.attempts: dict[str, list[float]] = {}
        with self.lock:
            legacy = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='owner'").fetchone()
            if legacy:
                row = self.db.execute('SELECT username,password_hash,must_change FROM owner').fetchone()
                if row and not self.db.execute('SELECT 1 FROM accounts WHERE id=1').fetchone():
                    self.db.execute('INSERT INTO accounts (id,username,password_hash,must_change) VALUES (1,?,?,?)', row)
                self.db.execute('DROP TABLE owner')
                self.db.commit()
            if not self.db.execute('SELECT 1 FROM accounts').fetchone():
                temporary = secrets.token_urlsafe(24)
                fd = os.open(self.bootstrap_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, 'w') as out:
                    out.write('OrthoGraph one-time setup credential\nUsername: ' + OWNER_USERNAME + '\nTemporary password: ' + temporary + '\nSet your own password on the sign-in page. This file is removed after setup.\n')
                self.bootstrap_path.chmod(0o600)
                self.db.execute('INSERT INTO accounts (id,username,password_hash,must_change) VALUES (1,?,?,1)', (OWNER_USERNAME, self.hasher.hash(temporary)))
                self.db.commit()

    def _rate_limited(self, key: str, limit: int, window: int) -> bool:
        now = time.time()
        recent = [t for t in self.attempts.get(key, []) if now - t < window]
        self.attempts[key] = recent
        return len(recent) >= limit

    def _record_attempt(self, key: str):
        self.attempts.setdefault(key, []).append(time.time())

    @staticmethod
    def _digest(token: str):
        return hashlib.sha256(token.encode()).hexdigest()

    def _new_session(self, account_id: int) -> str:
        token = secrets.token_urlsafe(32)
        self.db.execute('INSERT INTO sessions (token_hash,expires,account_id) VALUES (?,?,?)', (self._digest(token), int(time.time()) + SESSION_SECONDS, account_id))
        self.db.commit()
        return token

    def authenticate(self, username: str, password: str, new_password: str | None, client_key: str):
        with self.lock:
            key = 'login:' + client_key
            if self._rate_limited(key, 10, 900):
                return None, 'Too many attempts. Try again in 15 minutes.'
            row = self.db.execute('SELECT id,username,password_hash,must_change FROM accounts WHERE username=?', (username,)).fetchone()
            verified = False
            if row:
                try:
                    verified = self.hasher.verify(row[2], password)
                except (VerifyMismatchError, VerificationError):
                    pass
            if not verified:
                self._record_attempt(key)
                return None, 'Invalid username or password.'
            if row[3]:
                if new_password is None:
                    return None, 'SET_PASSWORD_REQUIRED'
                if not 12 <= len(new_password) <= 128 or new_password == password:
                    return None, 'Enter a new password of 12–128 characters, different from the temporary password.'
                self.db.execute('UPDATE accounts SET password_hash=?,must_change=0 WHERE id=?', (self.hasher.hash(new_password), row[0]))
                self.db.execute('DELETE FROM sessions WHERE account_id=?', (row[0],))
                if self.bootstrap_path.exists():
                    self.bootstrap_path.unlink()
            self.attempts.pop(key, None)
            return (self._new_session(row[0]), {'id': row[0], 'username': row[1]}), None

    def register(self, username: str, password: str, client_key: str):
        with self.lock:
            key = 'register:' + client_key
            if self._rate_limited(key, 5, 3600):
                return None, 'Too many registrations. Try again later.'
            self._record_attempt(key)
            if not USERNAME_PATTERN.fullmatch(username):
                return None, 'Use 3–32 characters: letters, numbers, dots, underscores or hyphens; start with a letter.'
            if not 12 <= len(password) <= 128:
                return None, 'Use a password of 12–128 characters.'
            if self.db.execute('SELECT COUNT(*) FROM accounts').fetchone()[0] >= MAX_ACCOUNTS:
                return None, 'This research demo has reached its account limit.'
            try:
                cursor = self.db.execute('INSERT INTO accounts (username,password_hash,must_change) VALUES (?,?,0)', (username, self.hasher.hash(password)))
                self.db.commit()
            except sqlite3.IntegrityError:
                return None, 'Username unavailable.'
            return (self._new_session(cursor.lastrowid), {'id': cursor.lastrowid, 'username': username}), None

    def user_for_token(self, token: str | None):
        if not token:
            return None
        with self.lock:
            row = self.db.execute('SELECT a.id,a.username FROM sessions s JOIN accounts a ON a.id=s.account_id WHERE s.token_hash=? AND s.expires>?', (self._digest(token), int(time.time()))).fetchone()
            return {'id': row[0], 'username': row[1]} if row else None

    def logout(self, token: str | None):
        if token:
            with self.lock:
                self.db.execute('DELETE FROM sessions WHERE token_hash=?', (self._digest(token),))
                self.db.commit()

    def needs_setup(self, account_id: int):
        with self.lock:
            row = self.db.execute('SELECT must_change FROM accounts WHERE id=?', (account_id,)).fetchone()
            return bool(row and row[0])

    def close(self):
        self.db.close()

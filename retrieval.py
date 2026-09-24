from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent


class ReferenceIndex:
    """Semantic retrieval over an explicitly curated, source-linked starter corpus."""
    def __init__(self, data_dir: Path):
        self.records = json.loads((ROOT/'references.json').read_text())
        for r in self.records:
            r['content_sha256'] = hashlib.sha256(r['text'].encode()).hexdigest()
        self.encoder = None
        self.client = None
        self.data_dir = data_dir
        self.lock = threading.RLock()
        self.status = 'not_loaded'

    def initialize(self):
        with self.lock:
            if self.client:
                return
            self.status = 'loading'
            self.encoder = TextEmbedding('BAAI/bge-small-en-v1.5',
                                         cache_dir=str(self.data_dir/'embedding_cache'), threads=2)
            client = QdrantClient(path=str(self.data_dir/'qdrant'))
            vectors = list(self.encoder.embed([r['text'] for r in self.records]))
            if not client.collection_exists('references'):
                client.create_collection('references', vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE))
            client.upsert('references', points=[models.PointStruct(
                id=str(uuid5(NAMESPACE_URL,r['id'])), vector=v.tolist(), payload=r
            ) for r,v in zip(self.records,vectors)])
            self.client = client
            self.status = 'ready'

    def search(self, query, topic=None, limit=3):
        with self.lock:
            self.initialize()
            vector = next(self.encoder.query_embed(query)).tolist()
            condition = models.Filter(must=[models.FieldCondition(
                key='topics', match=models.MatchValue(value=topic))]) if topic else None
            points = self.client.query_points('references', query=vector,
                                              query_filter=condition, limit=limit).points
            return [{**p.payload, 'retrieval_score':round(p.score,4)} for p in points]

    def attach(self, report):
        references, considerations = {}, []
        for f in report.findings:
            f.reference_ids = []
            if f.review == 'rejected':
                continue
            if not any(f.label in r['topics'] for r in self.records):
                continue
            hits = self.search(f'{f.title} panoramic radiograph limitations clinical assessment', f.label, 1)
            for hit in hits:
                references[hit['id']] = hit
                f.reference_ids.append(hit['id'])
                considerations.append({'finding_id':f.id, 'reference_id':hit['id'],
                                       'text':hit['consideration'], 'status':'review_context'})
        report.references = list(references.values())
        report.considerations = considerations
        return report

    def close(self):
        if self.client:
            self.client.close()

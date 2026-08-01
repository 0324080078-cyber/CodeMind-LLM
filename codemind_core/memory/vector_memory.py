"""Vector Memory using ChromaDB — fully local, no cloud."""

import os
import time
import uuid
from typing import List, Dict, Any, Optional


class VectorMemory:
    def __init__(self, persist_path="./memory/chroma_db", max_items=10000):
        self.persist_path = persist_path
        self.max_items = max_items
        self.client = None
        self.collection = None
        os.makedirs(persist_path, exist_ok=True)
        self._init()

    def _init(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=self.persist_path, settings=Settings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection(name="codemind_memory", metadata={"hnsw:space": "cosine"})
        except ImportError:
            print("ChromaDB not installed. Run: pip install chromadb")
        except Exception as e:
            print(f"Memory init failed: {e}")

    def store(self, content, session_id="global", metadata=None):
        if self.collection is None:
            return False
        try:
            meta = {"session_id": session_id, "timestamp": time.time()}
            if metadata:
                meta.update(metadata)
            self.collection.add(documents=[content], ids=[str(uuid.uuid4())], metadatas=[meta])
            return True
        except Exception as e:
            print(f"Memory store failed: {e}")
            return False

    def query(self, query, session_id=None, top_k=5):
        if self.collection is None:
            return []
        try:
            count = self.collection.count()
            if count == 0:
                return []
            where = {"session_id": session_id} if session_id else None
            results = self.collection.query(query_texts=[query], n_results=min(top_k, count), where=where)
            memories = []
            if results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    memories.append({"content": doc, "distance": results.get("distances", [[0]*len(results["documents"][0])])[0][i], "metadata": results.get("metadatas", [[{}]])[0][i]})
            return memories
        except Exception as e:
            print(f"Memory query failed: {e}")
            return []

    def forget(self, session_id):
        if self.collection is None:
            return False
        try:
            results = self.collection.get(where={"session_id": session_id})
            if results.get("ids"):
                self.collection.delete(ids=results["ids"])
            return True
        except Exception as e:
            print(f"Memory forget failed: {e}")
            return False

    def count(self):
        if self.collection is None:
            return 0
        return self.collection.count()

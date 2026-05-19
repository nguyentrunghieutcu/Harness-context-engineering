from enum import Enum
import time


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStore:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.memories = []

    def search(
            self,
            query: str,
            mtype=None,
            top_k: int = 5,
            min_sim: float = 0.12):
        return [m for m in self.memories if (
            mtype is None or m["type"] == mtype.value)][:top_k]

    def save(self, mtype: MemoryType, key: str, value: str, tags: str):
        entry = {
            "type": mtype.value,
            "key": key,
            "value": value,
            "tags": tags,
            "time": time.time()}
        self.memories.append(entry)
        return entry

    def delete(self, mtype: MemoryType, key: str):
        self.memories = [m for m in self.memories if not (
            m["type"] == mtype.value and m["key"] == key)]
        return True

    def list_keys(self, mtype=None, limit=30):
        return [m["key"] for m in self.memories if (
            mtype is None or m["type"] == mtype.value)][:limit]

    def evict_lru(self, keep_top: int):
        if len(self.memories) <= keep_top:
            return 0
        self.memories.sort(key=lambda x: x["time"], reverse=True)
        removed = len(self.memories) - keep_top
        self.memories = self.memories[:keep_top]
        return removed

    def stats(self):
        return {"total": len(self.memories)}


class EpisodicMemory:
    def __init__(self, store): self.store = store

    def save(
        self,
        key,
        value,
        tags): return self.store.save(
        MemoryType.EPISODIC,
        key,
        value,
        tags)


class SemanticMemory:
    def __init__(self, store): self.store = store

    def save(
        self,
        key,
        value,
        tags): return self.store.save(
        MemoryType.SEMANTIC,
        key,
        value,
        tags)


class ProceduralMemory:
    def __init__(self, store): self.store = store

    def save(
        self,
        key,
        value,
        tags): return self.store.save(
        MemoryType.PROCEDURAL,
        key,
        value,
        tags)

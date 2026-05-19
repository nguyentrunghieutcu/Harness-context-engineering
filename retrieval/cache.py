import time


class RetrievalCache:
    def __init__(self, ttl: int = 600):
        self.ttl = ttl
        self.cache = {}

    def _hash(self, query: str, paths: list[str]) -> str:
        return f"{query}:{','.join(sorted(paths))}"

    def get(self, query: str, paths: list[str]) -> list:
        key = self._hash(query, paths)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, query: str, paths: list[str], chunks: list):
        key = self._hash(query, paths)
        self.cache[key] = (chunks, time.time())

    def invalidate_all(self):
        self.cache.clear()

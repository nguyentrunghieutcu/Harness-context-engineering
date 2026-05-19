from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        tokenized_corpus = [c.content.split(" ") for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def score(self, query: str) -> list[float]:
        if not self.bm25:
            return []
        tokenized_query = query.split(" ")
        return self.bm25.get_scores(tokenized_query).tolist()

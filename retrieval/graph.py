import networkx as nx


class DependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, chunks: list):
        self.graph.clear()
        for c in chunks:
            self.graph.add_node(c.id, chunk=c)

        # Basic heuristic to build graph: look for symbol usage
        # In a real app, you would use tree-sitter references
        symbols = {c.symbol: c.id for c in chunks if c.symbol != "module"}

        for c in chunks:
            for sym, t_id in symbols.items():
                if sym in c.content and t_id != c.id:
                    self.graph.add_edge(c.id, t_id)  # c depends on sym

    def symbol_score(self, query: str, chunk) -> float:
        score = 0.0
        if chunk.symbol and chunk.symbol.lower() in query.lower():
            score += 0.5
        return score

    def expand(self, chunks: list, max_extra: int = 4) -> list:
        expanded = set()
        for c in chunks:
            if c.id in self.graph:
                neighbors = list(self.graph.successors(c.id)) + \
                    list(self.graph.predecessors(c.id))
                for n in neighbors:
                    expanded.add(n)

        new_chunks = []
        for n_id in expanded:
            if len(new_chunks) >= max_extra:
                break
            chunk_data = self.graph.nodes[n_id].get("chunk")
            if chunk_data and chunk_data not in chunks:
                new_chunks.append(chunk_data)

        return new_chunks

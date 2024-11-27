import networkx as nx

def max_acyclic_path(graph:nx.DiGraph, start_node):
    def dfs(node, visited):
        # Add current node to visited
        visited.add(node)

        # Initialize longest path and length
        max_length = 0
        max_path = []

        # Explore neighbors
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                path_length, path = dfs(neighbor, visited)
                if path_length > max_length:
                    max_length = path_length
                    max_path = path

        # Remove current node from visited to backtrack
        visited.remove(node)

        return max_length + 1, [node] + max_path

    return dfs(start_node, set())
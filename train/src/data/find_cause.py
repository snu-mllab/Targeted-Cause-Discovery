from collections import defaultdict
import numpy as np


class Graph:
    """ A class for generating cause matrix from a directed causal graph
    """

    def __init__(self, graph):
        """
        graph: numpy array (n x n). graph[i, j] = 1 if there is an edge from i to j
        """

        self.parent = defaultdict(set)  # dictionary containing parent List
        self.child = defaultdict(set)  # dictionary containing child List

        self.n = graph.shape[0]

        assert isinstance(graph, np.ndarray), "graph should be numpy array"
        self.graph = graph

        coord = np.nonzero(self.graph)
        for i, j in zip(coord[0], coord[1]):
            self.addEdge(i, j)

    def addEdge(self, i, j):
        self.child[i].add(j)
        self.parent[j].add(i)

    def mergeParents(self, i):
        cause = self.parent[i]
        for j in self.parent[i]:
            cause = cause | self.parent[j]

        self.parent[i] = cause

    def topologicalSortUtil(self, i, visited, stack):
        visited[i] = True

        for j in self.child[i]:
            if visited[j] == False:
                self.topologicalSortUtil(j, visited, stack)

        stack.insert(0, i)

    def topologicalSort(self):
        visited = [False] * self.n
        stack = []

        for i in range(self.n):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        return stack

    def causeMatrix(self):
        stack = self.topologicalSort()

        for i in stack:
            self.mergeParents(i)

        cause_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            cause_matrix[list(self.parent[i]), i] = 1.

        return cause_matrix

    def causeCount(self):
        stack = self.topologicalSort()
        stack_reverse = stack[::-1]

        dist_matrix = np.zeros((self.n, self.n), dtype=np.int32)
        self.parent_prev = defaultdict(set)
        self.parent_next = defaultdict(set)

        for i in range(self.n):
            self.parent_next[i] = self.parent[i].copy()

        for k in range(1, self.n):
            total = 0
            for i in stack_reverse:  # start from root to leaves
                dist_matrix[list(self.parent_next[i]), i] = k
                self.parent_prev[i] = self.parent_prev[i] | self.parent_next[i]

                next_cause = set()
                for j in self.parent_next[i]:
                    next_cause = next_cause | self.parent[j]

                self.parent_next[i] = set()
                for j in next_cause:
                    if j not in self.parent_prev[i]:  # shortest path
                        self.parent_next[i].add(j)

                total += len(self.parent_next[i])

            if total == 0:
                break

        return dist_matrix


if __name__ == "__main__":
    import os

    idx = 10
    p = 100

    for idx in range(1, 1 + idx):
        base_path = "/storage/janghyun/datasets/causal"
        name = f"train_synthetic_SF/data_p{p}_e{2*p}_n10000_linearSF_noise0.5"
        fp_graph = os.path.join(os.path.join(base_path, name), f"DAG{idx}.npy")
        graph = np.load(fp_graph)
        # print(graph)

        cause_matrix = Graph(graph).causeMatrix()
        cause_matrix2 = Graph(cause_matrix).causeMatrix()
        print(np.abs(cause_matrix - cause_matrix2).sum())

        dist_matrix = Graph(graph).causeCount()
        print(np.abs((dist_matrix > 0) - cause_matrix2).sum())
        print(dist_matrix[graph > 0].max() == 1.0, dist_matrix[graph > 0].min() == 1.0)

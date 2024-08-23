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

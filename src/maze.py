import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import heapq


class Maze:
    def __init__(self, data):
        self.data = data
        self.start = 0
        self.end = data.shape[0] * data.shape[0] - 1

    def incidence_matrix(self):
        n, m = self.data.shape
        A = sparse.lil_matrix((n * n, n * n))

        for i in range(n):
            for j in range(n):
                idx = i * m + j

                if i > 0 and self.data[i - 1, j] == 0:  # upper neighbour
                    A[idx, idx - n] = 1

                if i < n - 1 and self.data[i + 1, j] == 0:  # lower neighbour
                    A[idx, idx + n] = 1

                if j > 0 and self.data[i, j - 1] == 0:  # left neighbour
                    A[idx, idx - 1] = 1

                if j < m - 1 and self.data[i, j + 1] == 0:  # right neighbour
                    A[idx, idx + 1] = 1

                A[idx, idx] = 0
        return A

    def dijkstra(self, incidence_matrix):
        n, m = self.data.shape
        nodes = n * n
        distances = np.full(nodes, np.inf)
        visited = np.full(nodes, False)

        distances[self.start] = 0
        priority_queue = [(0, self.start)]
        previous = np.full(nodes, -1)

        while priority_queue:
            d, u = heapq.heappop(priority_queue)
            if not visited[u]:
                visited[u] = True

                if u == self.end:
                    break

                for v in range(nodes):
                    if incidence_matrix[u, v] == 1:
                        alt = distances[u] + 1
                        if alt < distances[v]:
                            distances[v] = alt
                            previous[v] = u
                            heapq.heappush(priority_queue, (alt, v))

        return previous

    def find_shortest_path(self):
        incidence_matrix = self.incidence_matrix()
        previous = self.dijkstra(incidence_matrix)

        path = []
        u = self.end
        while u != -1:
            path.append(u)
            u = previous[u]
        path.reverse()

        return path

    def draw_maze(self):
        plt.figure()
        plt.imshow(self.data, cmap="binary")
        plt.show()


def load_maze_csv(file_name):
    data = np.genfromtxt(file_name, delimiter=",")
    return data.astype(bool)


def generate_maze():
    pass


data = load_maze_csv("./data/maze_5.csv")
print(data)
maze = Maze(data)
maze.draw_maze()

indicence_matrix = maze.incidence_matrix()
shortest_path = maze.find_shortest_path()
print(shortest_path)

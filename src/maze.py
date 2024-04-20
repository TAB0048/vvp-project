import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.sparse as sparse
import heapq


class Maze:
    def __init__(self, data=None):
        self.data = data
        if data is None:
            self.start = None
            self.end = None
        else:
            self.start = 0
            self.end = data.shape[0] * data.shape[0] - 1

    def load_maze_csv(self, file_name):
        data = np.genfromtxt(file_name, delimiter=",")
        self.data = data.astype(bool)
        self.start = 0
        self.end = data.shape[0] * data.shape[0] - 1

    def save_maze_csv(self, file_name):
        np.savetxt(file_name, self.data.astype(int), delimiter=",", fmt="%i")

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

        if previous[u] != -1:
            while u != -1:
                path.append(u)
                u = previous[u]
            path.reverse()

        return path

    def draw_maze(self):
        plt.figure()
        plt.imshow(self.data, cmap="binary")
        plt.show()

    def draw_maze_path(self):
        plt.figure()
        plt.imshow(self.data, cmap="binary")  # maze

        shortest_path = self.find_shortest_path()

        if shortest_path == []:
            print("Path doesn't exist.")
            return

        path_matrix = np.zeros_like(self.data)
        row, col = np.divmod(shortest_path, self.data.shape[0])
        path_matrix[row, col] = 1

        path_colors = ListedColormap([(0, 0, 0, 0), "red"])
        plt.imshow(path_matrix, cmap=path_colors)  # path

        plt.show()


class MazeTemplate:
    @staticmethod
    def empty(n):
        return Maze(np.zeros((n, n), dtype=bool))

    @staticmethod
    def slalom(n):
        maze = MazeTemplate.empty(n)
        idx = n//5
        w = n//10  # width
        maze.data[0:-idx, idx - w:idx + w] = True
        maze.data[-idx - 2*w:-idx, idx - w:-2*idx] = True

        maze.data[idx:, -idx - w:-idx + w] = True
        maze.data[idx: idx + 2*w, 2*idx: -idx + w] = True
        return maze

    @staticmethod
    def random_obstacles(template, max_obstacles):
        tmp_maze = Maze(np.copy(template.data))
        n, m = template.data.shape

        for i in range(max_obstacles):
            x = np.random.randint(0, n)
            y = np.random.randint(0, n)

            if not tmp_maze.data[x, y]:
                tmp_maze.data[x, y] = True
                path = tmp_maze.find_shortest_path()

                if len(path) > 0:
                    template.data[x, y] = True

        return template


new_maze = MazeTemplate.random_obstacles(MazeTemplate.slalom(20), 50)
new_maze.draw_maze()
new_maze.save_maze_csv("new_maze.csv")

maze = Maze()
maze.load_maze_csv("new_maze.csv")
maze.draw_maze()
maze.draw_maze_path()

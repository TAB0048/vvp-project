import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


class Maze:
    def __init__(self, data):
        self.data = data

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

    def bfs(self):
        pass

    def find_shortest_path(self):
        pass

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
print(indicence_matrix)

plt.figure()
plt.imshow(indicence_matrix.todense(), cmap="binary")
plt.show()

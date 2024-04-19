import numpy as np
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, data):
        self.data = data

    def incidence_matrix(self):
        pass

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

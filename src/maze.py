import numpy as np


class Maze:
    def __init__(self, data):
        self.data = data

    def incidence_matrix(self):
        pass

    def bfs(self):
        pass

    def find_shortest_path(self):
        pass


def load_maze_csv(file_name):
    data = np.genfromtxt(file_name, delimiter=",")
    return data.astype(bool)


def generate_maze():
    pass


data = load_maze_csv("./data/maze_5.csv")
print(data)

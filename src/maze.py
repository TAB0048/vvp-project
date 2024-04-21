import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.sparse as sparse
import heapq


class Maze:
    """
    Class representing a maze.

    It contains maze data, it's start position (upper left)
    and end position (lower right).

    It allows user to load data from and save as CSV file, generate incidence
    matrix and find shortest path from start to end.

    """
    def __init__(self, data: np.ndarray | None = None) -> None:
        """
        Constructor of the maze object.

        Args:
            data (np.ndarray | None): he initial data for maze.

        Returns:
            None
        """
        self.data = data
        if data is None:
            self.start = None
            self.end = None
        else:
            self.start = 0
            self.end = data.shape[0] * data.shape[0] - 1

    def load_maze_csv(self, file_name: str) -> None:
        """
        Loads maze data from a CSV file.

        Args:
            file_name (str): The path to the CSV file with maze data.

        Returns:
            None
        """
        data = np.genfromtxt(file_name, delimiter=",")
        self.data = data.astype(bool)
        self.start = 0
        self.end = data.shape[0] * data.shape[0] - 1

    def save_maze_csv(self, file_name: str) -> None:
        """
        Saves the maze as a CSV file.

        Args:
            file_name (str): The name of the new CSV file.

        Returns:
            None
        """
        np.savetxt(file_name, self.data.astype(int), delimiter=",", fmt="%i")

    def incidence_matrix(self) -> sparse.lil_matrix:
        """
        Generates an incidence matrix of the maze.

        Args:
            None

        Returns:
            sparse.lil_matrix: The incidence matrix of the maze.
        """
        n, m = self.data.shape
        A = sparse.lil_matrix((n * n, n * n))

        for i in range(n):
            for j in range(n):
                idx = i * m + j

                if i > 0 and self.data[i - 1, j] == 0:  # upper neighbour
                    A[idx, idx - n] = 1 if self.data[i, j] == 0 else 0

                if i < n - 1 and self.data[i + 1, j] == 0:  # lower neighbour
                    A[idx, idx + n] = 1 if self.data[i, j] == 0 else 0

                if j > 0 and self.data[i, j - 1] == 0:  # left neighbour
                    A[idx, idx - 1] = 1 if self.data[i, j] == 0 else 0

                if j < m - 1 and self.data[i, j + 1] == 0:  # right neighbour
                    A[idx, idx + 1] = 1 if self.data[i, j] == 0 else 0

                A[idx, idx] = 0
        return A

    def dijkstra(self, incidence_matrix: sparse.lil_matrix) -> np.ndarray:
        """
        Performs the Dijkstra's algorithm to find the shortest path
        in the maze.

        Args:
            indicence_matrix (sparse.lil_matrix): The incidence matrix
                                                  of the maze.

        Returns:
            np.ndarray: An array containing cells of the shortest path.
        """
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

    def find_shortest_path(self) -> list:
        """
        Finds the shortest path from upper left corner to lower right corner
        of the maze using incidence matrix and Dijkstra's algorithm.

        Args:
            None

        Returns:
            list: A list with incides representing the shortest path.
        """
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

    def draw_maze(self) -> None:
        """
        Creates a picture of the maze only (without any path).
        """
        plt.figure()
        plt.imshow(self.data, cmap="binary")
        plt.show()

    def draw_maze_path(self) -> None:
        """
        Creates a picture of the maze, found shortest path included.
        """
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
    """
    Class representing templates for mazes.

    It allows user to generate mazes from templates and add obstacles to it.
    """
    @staticmethod
    def empty(n: int) -> Maze:
        """
        Creates an empty maze with given size.

        Args:
            n (int): The size of the created maze.

        Returns:
            Maze: An empty maze with size n x n.
        """
        return Maze(np.zeros((n, n), dtype=bool))

    @staticmethod
    def slalom(n: int) -> Maze:
        """
        Creates a maze with two L-shaped obstacles (so that the path
        is at least S-shaped).

        Args:
            n (int): The size of the created maze.

        Returns:
            Maze: A new maze with size n x n and L-shaped obstacles.
        """
        maze = MazeTemplate.empty(n)
        idx = n//5
        w = n//10  # width
        maze.data[0:-idx, idx - w:idx + w] = True
        maze.data[-idx - 2*w:-idx, idx - w:-2*idx] = True

        maze.data[idx:, -idx - w:-idx + w] = True
        maze.data[idx: idx + 2*w, 2*idx: -idx + w] = True
        return maze

    @staticmethod
    def random_obstacles(template: Maze, max_obstacles: int) -> Maze:
        """
        Generates random obstacles to existing maze template and checks
        if maze still has solution (path).

        Args:
            template (Maze): The maze we add obstacles to.
            max_obstacles (int): The maximum amount of new obstacles.

        Returns:
            Maze: Modified maze with new obstacles.
        """
        tmp_maze = Maze(np.copy(template.data))
        n, m = template.data.shape
        obstacles = 0

        while obstacles < max_obstacles:
            x = np.random.randint(0, n)
            y = np.random.randint(0, n)

            if not tmp_maze.data[x, y]:
                tmp_maze.data[x, y] = True
                path = tmp_maze.find_shortest_path()

                if len(path) > 0:
                    template.data[x, y] = True
                    obstacles += 1
                else:
                    tmp_maze.data[x, y] = False

        return template

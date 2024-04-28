"""
Module to define maze templates.
"""

import numpy as np
from .maze import Maze


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
    def lines(n: int) -> Maze:
        """
        Creates a maze with line-shaped obstacles, each line with only one cell
        to go through.

        Args:
            n (int): The size of the created maze.

        Returns:
            Maze: A new maze with size n x n and line-shaped obstacles.
        """
        maze = MazeTemplate.empty(n)

        maze.data[1:n:4] = 1
        v = np.random.randint(0, n, n//4)
        for row, col in enumerate(v):
            maze.data[4 * row + 1, col] = 0

        maze.data[n - 1, n - 1] = 0
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

        for i in range(max_obstacles):
            x = np.random.randint(0, n)
            y = np.random.randint(0, n)

            if not tmp_maze.data[x, y]:
                tmp_maze.data[x, y] = True
                path = tmp_maze.find_shortest_path()

                if len(path) > 0:
                    template.data[x, y] = True
                else:
                    tmp_maze.data[x, y] = False

        return template

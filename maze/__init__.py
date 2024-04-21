"""
This is a maze package.

It contains following modules:
- maze: contains a Maze class
- maze_templates: containcs a MazeTemplate class
"""

from .maze import Maze
from .maze_templates import MazeTemplate

__all__: list[str] = ["Maze", "MazeTemplate"]

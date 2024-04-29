# Finding the shortest path in a maze

## Description

This project is about solving and generating mazes. The main input is a maze $n\times n$, where the start is the upper left corner and the end is the bottom right corner. It is only possible to get from one cell to another through an edge (not through a corner). This project implements algorithms for loading mazes, finding shortest path and generating mazes.

Loaded maze is saved as a NumPy matrix with True/False values (True = cell can't be passed-through).
The output, as a picture, contains three colors of cells: 
- black = can't pass through 
- white = can pass through 
- red = the shortest path

## Functionalities

- loading a maze from a CSV file
- finding the shortest path from start to end in the following two steps:
  - creating an incidence matrix
  - finding the shortest path using Dijkstra's algorithm
- draw the maze as a black-and-white picture, with or without solution (colored red)
- generating a new maze (where the shortest path exists):
  - there are three available templates: empty, slalom and lines
  - obstacles can be added to the predefined templates

## Examples

Examples of using mentioned functionalities are demonstrated in the examples.ipynb file.

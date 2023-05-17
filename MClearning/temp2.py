import tkinter as tk
import numpy as np
import random


MAZE_H = 5  # grid height
MAZE_W = 5  # grid width
UNIT = 40   # pixels(40,4,4)

# origin = np.array([20 + np.random.randint(0, MAZE_H - 1) * UNIT,
#                                20 + np.random.randint(0, MAZE_W - 1) * UNIT])
#
# print(origin)
origin = np.array([0, 1])
hell_centers = [np.array([0, 0]), np.array([0, 1]), np.array([0, 2]), np.array([1, 0]), np.array([1, 1]), np.array([1, 2])]
for i in range(10):
    origin = np.array([0, 1])
    while np.any(np.all(origin == hell_centers, axis=1)):
        origin = np.array([random.randint(0, 2), random.randint(0, 2)])
    print('\r', origin)
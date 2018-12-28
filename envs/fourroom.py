import gym
from gym.spaces import Discrete
import math
import numpy as np


class FourRoomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    WALL_TILE = 1
    FREE_TILE = 0

    def __init__(self):

        # action-space
        self._action_set = [[-1, 0], [0, -1], [1, 0], [0, 1]]
        self.action_space = Discrete(len(self._action_set))

        self.room_size = 5

        #  derived properties
        self.grid_size = self.room_size * 2 + 3
        self.states_count = self.grid_size ** 2

        self.grid = self._generate_simple_grid()

        self.position = [1, 1]
        self.reset()

    def _generate_simple_grid(self):
        grid = []
        total_size = self.room_size * 2 + 3
        door_position = math.ceil(self.room_size / 2)

        for index in range(total_size):
            if index == 0 or index == total_size - 1:
                grid.append([1] * total_size)
            else:
                row = [0] * total_size
                row[0] = self.WALL_TILE
                row[-1] = self.WALL_TILE
                row[self.room_size + 1] = self.WALL_TILE

                if index == self.room_size + 1:
                    row = [1] * total_size

                if index == door_position or index == total_size - door_position - 1:
                    row[self.room_size + 1] = self.FREE_TILE
                    row[self.room_size + 1] = self.FREE_TILE

                row[door_position] = self.FREE_TILE
                row[-door_position - 1] = self.FREE_TILE

                grid.append(row)

        return grid

    def _state_to_position(self, state):
        row = math.floor(state / self.grid_size)
        col = state % self.grid_size
        return row, col

    def _position_to_state(self, position):
        row = position[0]
        col = position[1]

        return row * self.grid_size + col

    def reset(self):
        self.position = [1, 1]
        return self._position_to_state(self.position)

    def step(self, action):

        action_pos = self._action_set[action]
        new_position = np.add(self.position, action_pos)

        try:
            if self.grid[new_position[0]][new_position[1]] is self.FREE_TILE:
                self.position = new_position
        except:
            # position out or range
            self.position = self.position

        return self._position_to_state(self.position), 0, False, {}
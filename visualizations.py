import copy
import matplotlib.pyplot as plt
import numpy as np


class Visualizations:

    def __init__(self, env):
        self.env = env

    def visualize_subgoals(self, subgoal_states):

        grid_with_subgoals = copy.deepcopy(self.env.grid)

        for goal_state in subgoal_states:
            goal = self.env._state_to_position(goal_state)
            grid_with_subgoals[goal[0]][goal[1]] = 10

        plt.title("Subgoal visualization")
        plt.imshow(grid_with_subgoals)
        plt.show()

    def visualize_sr_state(self, sr):

        result = np.zeros((self.env.grid_size, self.env.grid_size))

        for state, value in enumerate(sr):
            position = self.env._state_to_position(state)
            result[position[0]][position[1]] = value

        plt.title("SR for state")
        plt.imshow(result)
        plt.show()

    def _build_reward_map(self, sr):

        map = np.zeros(shape=(self.env.grid_size, self.env.grid_size))

        for state in range(self.env.states_count):

            position = self.env._state_to_position(state)
            # do not test state if it is a wall
            if self.env.grid[position[0]][position[1]] == self.env.WALL_TILE:
                continue

            for action in self.env._action_set:
                result_position = np.add(position, action)
                result_state = self.env._position_to_state(result_position)

                reward = sr[result_state] - sr[state]

                current_value = map[position[0]][position[1]]
                map[position[0]][position[1]] = np.maximum(reward, current_value)

        return map

    def visualize_subgoal_reward_map(self, sr):

        map = self._build_reward_map(sr)

        plt.figure(figsize=(10, 10))
        plt.imshow(map)
        plt.title("Reward map for subgoal")

        for row in range(len(map)):
            for col in range(len(map[0])):

                if self.env.grid[row][col] is not self.env.WALL_TILE:
                    plt.text(y=row, x=col, s=round(map[row][col], 3), color='w', ha="center", va="center")

        plt.show()

    def visualize_policy(self, q, subgoal_state, action_meaning, id):

        grid = copy.deepcopy(self.env.grid)

        policy = np.zeros(shape=(self.env.grid_size, self.env.grid_size), dtype=int)

        # get the index of the highest action-value for each state
        for state, action_values in enumerate(q):
            position = self.env._state_to_position(state)
            policy[position[0]][position[1]] = np.argmax(action_values)

        # give the subgoal state a color on the map
        subgoal_position = self.env._state_to_position(subgoal_state)
        grid[subgoal_position[0]][subgoal_position[1]] = 3

        plt.imshow(grid)

        for row in range(len(policy)):
            for col in range(len(policy[0])):

                if self.env.grid[row][col] is not self.env.WALL_TILE:
                    plt.text(y=row, x=col, s=action_meaning[policy[row][col]], color='w', ha="center", va="center")

        plt.title(f"Learned policy for {id}")
        plt.show()


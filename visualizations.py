import copy
import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizations:

    def __init__(self, env, data_dir):
        self.env = env
        self.data_dir = data_dir

        self.show_plots = False
        self.save_plots = True

    def visualize_subgoals(self, subgoal_states, iteration):

        plt.figure()

        grid_with_subgoals = copy.deepcopy(self.env.grid)

        for goal_state in subgoal_states:
            goal = self.env._state_to_position(goal_state)
            grid_with_subgoals[goal[0]][goal[1]] = 10

        plt.title(f"Subgoal visualization iteration: {iteration}")
        plt.imshow(grid_with_subgoals)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, f"subgoals_{iteration}.png"))

        plt.clf()
        plt.close()

    def visualize_sr_state(self, sr, state, iteration):

        plt.figure()

        result = np.zeros((self.env.grid_size, self.env.grid_size))

        for state, value in enumerate(sr):
            position = self.env._state_to_position(state)
            result[position[0]][position[1]] = value

        plt.figure(figsize=(self.env.grid_size / 2, self.env.grid_size / 2))
        plt.title(f"SR for state {state} iteration {iteration}")
        plt.imshow(result)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, f"sr_state_{state}_{iteration}.png"))

        plt.clf()
        plt.close()

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

    def visualize_candidate_subgoals(self, states, iteration):

        plt.figure()

        # visualize all candidate subgoals
        grid_copy = copy.deepcopy(self.env.grid)

        for candidate in states:
            position = self.env._state_to_position(candidate)
            grid_copy[position[0]][position[1]] = 2

        plt.clf()
        plt.imshow(grid_copy)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, f"candidate_subgoals_{iteration}.png"))

        plt.clf()
        plt.close()

    def visualize_env(self):

        plt.figure()
        plt.title("Environment grid")
        plt.imshow(self.env.grid)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, "env.png"))

        plt.clf()
        plt.close()

    def visualize_policy_learning_curve(self, history):

        plt.figure(figsize=(5, 5))
        plt.title("Policy learning curve")
        plt.plot(history)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, "learning_curve.png"))

        plt.clf()
        plt.close()

    def visualize_sr(self, sr):

        plt.figure()
        plt.title("SR visualization")
        plt.imshow(sr)

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, "sr.png"))

        plt.clf()
        plt.close()

    def visualize_subgoal_reward_map(self, sr, subgoal_state, iteration):

        map = self._build_reward_map(sr)

        plt.figure(figsize=(self.env.grid_size / 2, self.env.grid_size / 2))
        plt.imshow(map)
        plt.title(f"Reward map for subgoal state {subgoal_state} iteration {iteration}")

        for row in range(len(map)):
            for col in range(len(map[0])):

                if self.env.grid[row][col] is not self.env.WALL_TILE:
                    plt.text(y=row, x=col, s=round(map[row][col], 3), color='w', ha="center", va="center")

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, f"subgoal_reward_map_{subgoal_state}_{iteration}.png"))

        plt.clf()
        plt.close()

    def visualize_mutliple_learning_curves(self, history):

        plt.figure(figsize=(5, 5))

        for curve in history:
            plt.plot(curve)

        plt.ylim(0, 1000)
        plt.xlim(0, 100)
        plt.show()

        plt.clf()
        plt.close()

    def visualize_policy(self, q, start_state, goal_state, action_meaning, id):

        grid = copy.deepcopy(self.env.grid)

        plt.figure(figsize=(self.env.grid_size/2, self.env.grid_size/2))

        policy = np.zeros(shape=(self.env.grid_size, self.env.grid_size), dtype=int)

        # get the index of the highest action-value for each state
        for state, action_values in enumerate(q):
            position = self.env._state_to_position(state)
            policy[position[0]][position[1]] = np.argmax(action_values)

        for row in range(len(policy)):
            for col in range(len(policy[0])):

                if self.env.grid[row][col] != self.env.WALL_TILE:
                    plt.text(y=row, x=col, s=action_meaning[policy[row][col]], color='w', ha="center", va="center")

        # give the goal state a color on the map
        goal_position = self.env._state_to_position(goal_state)
        grid[goal_position[0]][goal_position[1]] = 3
        plt.text(y=goal_position[0], x=goal_position[1], s="G", color='w', ha="center", va="center")

        # give the starting state also a color
        if start_state is not None:
            start_position = self.env._state_to_position(start_state)
            grid[start_position[0]][start_position[1]] = 5

        plt.imshow(grid)

        plt.title(f"Learned policy {id}")

        if self.show_plots:
            plt.show()

        if self.save_plots:
            plt.savefig(os.path.join(self.data_dir, f"policy_{id}.png"))

        plt.clf()
        plt.close()

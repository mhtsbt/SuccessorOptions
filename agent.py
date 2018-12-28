import argparse
import gym
import envs
from tqdm import tqdm
import numpy as np
import random
from visualizations import Visualizations
import scipy
from sklearn.cluster import KMeans
import os
import shutil

class SuccessorOptionsAgent:

    def __init__(self, env_name, alpha, gamma, rollout_samples, options_count, option_learning_steps):
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make(env_name)
        self.rollout_samples = rollout_samples
        self.option_learning_steps = option_learning_steps
        self.options_count = options_count
        self.viz = Visualizations(env=self.env)

    @staticmethod
    def _save_array(array, filename):
        np.save(os.path.join(DATA_DIR, filename), array)

    @staticmethod
    def _load_array(filename):
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            return None

        return np.load(path)

    @staticmethod
    def _log(msg):
        print(msg)

    def update_sr(self, sr, state, prev_state):

        # a one-hot vector with all zeros except a 1 at the s-th position
        indicator = np.zeros(self.env.states_count)
        indicator[prev_state] = 1

        # eq3 (TD-update)
        sr[prev_state] += self.alpha * (indicator + self.gamma * sr[state] - sr[prev_state])

        return sr

    def cluster_sr(self, sr):
        kmeans = KMeans(n_clusters=self.options_count, random_state=0).fit(sr)
        clusters = kmeans.cluster_centers_

        return clusters

    def get_subgoal_states(self, sr, sr_clusters):

        subgoals_states = []

        distances_to_centers = np.zeros((self.env.states_count, len(sr_clusters)))
        for state, srs in enumerate(sr):

            for i in range(len(sr_clusters)):
                distances_to_centers[state][i] = scipy.spatial.distance.cosine(sr_clusters[i], srs)

        for i in range(len(sr_clusters)):
            state = np.nanargmin(distances_to_centers[:, i])
            subgoals_states.append(state)

        return subgoals_states

    def run_policy(self, eps, sr, steps):

        new_sr = np.zeros(shape=(self.env.states_count, self.env.states_count))

        # go back to the starting position
        prev_state = self.env.reset()

        test = np.array([])

        for _ in tqdm(range(steps)):

            # take an action in the environment
            action = random.randint(0, 3)
            state, _, _, _ = self.env.step(action)

            # update the SR in a TD-style
            new_sr = self.update_sr(new_sr, state, prev_state)

            prev_state = state

        return new_sr

    def learn_option_q(self, sr, subgoal_state, steps):

        lr = 0.25
        gamma = 0.99
        eps = 0.3

        # initialize all Q-values to zero
        q = np.zeros((self.env.states_count, self.env.action_space.n))

        prev_state = self.env.reset()

        for _ in tqdm(range(steps)):

            if random.random() < eps:
                # take random actions in the environment
                action = random.randint(0, self.env.action_space.n-1)
            else:
                # take greedy action
                action = np.random.choice(np.flatnonzero(q[prev_state] == q[prev_state].max()))

            state, _, _, _ = self.env.step(action)

            # use the learned SR of the subgoal-state as intrinsic reward
            reward = sr[state]-sr[prev_state]

            # update the q-value (in a TD-fashion)
            best_future_value = np.max(q[state])
            q[prev_state][action] += lr*(reward+gamma*best_future_value-q[prev_state][action])

            if state == subgoal_state:
                # the goal was found
                prev_state = self.env.reset()
            else:
                # not final state, continue
                prev_state = state

        return q

    @staticmethod
    def smdp_td_update(q, state, prev_state, action, reward):

        lr = 0.25
        gamma = 0.99

        best_future_value = np.max(q[state])
        q[prev_state][action] += lr*(reward+gamma*best_future_value-q[prev_state][action])

    def run_smdp(self, option_policies, goal_state, subgoal_states, episodes):

        eps = 0.1
        action_option_dist = 1 # 1 = more actions, 0 = only options

        # learn option-policies that lead to the subgoal-states
        # this q-table also contains values for running primary actions options [0-n][prim actions]
        q = np.zeros(shape=(self.env.grid_size, self.env.grid_size, self.env.action_space.n + self.options_count))

        for _ in range(episodes):

            prev_state = self.env.reset()

            for episode_step in range(int(1e6)):

                if random.random() < eps:
                    # do something random

                    if random.random() < action_option_dist:
                        # follow random action
                        action = random.randint(self.options_count-1, self.env.action_space.n - 1)
                        state, _, _, _ = self.env.step(action)

                        if state == goal_state:
                            reward = 1
                        else:
                            reward = 0

                        self.smdp_td_update(q, state, prev_state, action, reward)

                    else:
                        # follow random option
                        option = random.randint(0, self.options_count-1)

                        # TODO: follow option
                else:
                    # greedy actions
                    greedy_action = np.random.choice(np.flatnonzero(q[prev_state] == q[prev_state].max()))

                    if greedy_action < self.options_count:
                        # use option
                        sel_option = greedy_action
                    else:
                        # use primary action
                        sel_action = greedy_action - self.options_count-1
                        state, _, _, _ = self.env.step(sel_action)



                prev_state = state

    def run(self, iterations):

        initial_sr_filename = "initial_sr.npy"

        for _ in range(iterations):
            # if we did this before, re-use the SR
            sr = self._load_array(initial_sr_filename)

            if sr is None:
                # run the policy (completely random) for the first time
                sr = self.run_policy(eps=1, sr=None, steps=self.rollout_samples)

                # save the result, so next time no need to do this again
                self._save_array(sr, initial_sr_filename)

            # do the clustering, to find the subgoals
            sr_clusters = self.cluster_sr(sr)
            subgoal_states = self.get_subgoal_states(sr, sr_clusters)

            # visualize each of the subgoals SR
            for subgoal_state in subgoal_states:
                state_sr = sr[subgoal_state]
                self.viz.visualize_sr_state(state_sr)
                self.viz.visualize_subgoal_reward_map(state_sr)

            self.viz.visualize_subgoals(subgoal_states)

            option_policies_filename = "option_policies.npy"
            option_policies = self._load_array(option_policies_filename)

            if option_policies is None:

                option_policies = []

                for subgoal_state in subgoal_states:
                    state_sr = sr[subgoal_state]
                    option_q = self.learn_option_q(sr=state_sr, subgoal_state=subgoal_state, steps=self.option_learning_steps)
                    option_policies.append(option_q)

                self._save_array(option_policies, option_policies_filename)

            # visualize the learned or the loaded policies for the options
            for option_id, subgoal_state in enumerate(subgoal_states):
                self.viz.visualize_policy(option_policies[option_id], subgoal_state)

            # run the SMDP
            self.run_smdp(option_policies=option_policies, goal_state=subgoal_states[3], subgoal_states=subgoal_states)


            print(subgoal_states)


# cli arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--gamma', default=0.95)
parser.add_argument('--reset', default=False)
parser.add_argument('--options_count', default=4)
parser.add_argument('--iterations', default=1)
parser.add_argument('--env', default="FourRoom-v0")
parser.add_argument('--rollout_samples', default=int(5e6))
parser.add_argument('--option_learning_steps', default=int(1e6))
DATA_DIR = "data"

args = parser.parse_args()

if args.reset:
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

so = SuccessorOptionsAgent(env_name=args.env,
                           alpha=args.alpha,
                           gamma=args.gamma,
                           rollout_samples=args.rollout_samples,
                           options_count=args.options_count,
                           option_learning_steps=args.option_learning_steps)

so.run(iterations=args.iterations)

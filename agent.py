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
import matplotlib.pyplot as plt


class SuccessorOptionsAgent:

    def __init__(self, env_name, alpha, gamma, rollout_samples, options_count, option_learning_steps):
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make(env_name)
        self.rollout_samples = rollout_samples
        self.option_learning_steps = option_learning_steps
        self.options_count = options_count
        self.viz = Visualizations(env=self.env, data_dir=DATA_DIR)

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

    # this is used for policy visualization (a map of what each action-index means (up, down, option1, ...)
    def get_action_meaning_smdp(self):

        option_meaning = []

        for i in range(self.options_count):
            option_meaning.append("O"+str(i+1))

        action_meaning = self.env._action_meaning
        return action_meaning + option_meaning

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

    # The candidate states are those states which have a moderately developed SR
    @staticmethod
    def get_candidate_subgoals(sr):

        sr_sums = np.array([sum(row) for row in sr])

        sr_max, sr_min = np.percentile(sr_sums, [75, 25])
        valid_states = np.argwhere((sr_sums > sr_min) & (sr_sums < sr_max))

        return valid_states.reshape(len(valid_states))

    def get_subgoal_states(self, sr, sr_clusters, canidate_subgoals):

        subgoals_states = []

        distances_to_centers = np.zeros((self.env.states_count, len(sr_clusters)))
        for state, srs in enumerate(sr):

            if state not in canidate_subgoals:
                # if the state is not a candidate set all distances to 1
                distances_to_centers[state] = np.ones(shape=(len(sr_clusters)))
            else:
                # if it is a valid candidate subgoal, calculate the distance
                for i in range(len(sr_clusters)):
                    distances_to_centers[state][i] = scipy.spatial.distance.cosine(sr_clusters[i], srs)

        for i in range(len(sr_clusters)):
            state = np.nanargmin(distances_to_centers[:, i])
            subgoals_states.append(state)

        return subgoals_states

    def run_random_policy(self, steps):

        new_sr = np.zeros(shape=(self.env.states_count, self.env.states_count))

        # go back to the starting position
        prev_state = self.env.reset()

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

    def run_smdp(self, option_policies, goal_state, start_state, subgoal_states, episodes, action_option_sampling):

        eps = 0.1
        history = []
        max_episode_steps = int(1e6)

        # prepare the SR
        new_sr = np.zeros(shape=(self.env.states_count, self.env.states_count))

        # learn option-policies that lead to the subgoal-states
        # this q-table also contains values for running primary actions options [prim actions][options]
        q = np.zeros(shape=(self.env.states_count, self.env.action_space.n + self.options_count))

        cum_reward = 0

        for ep in range(episodes):

            prev_state = self.env.reset(start_state=start_state)
            episode_step = 0

            while True:

                # stop the loop on max allowed steps
                if episode_step > max_episode_steps:
                    self._log(f"Failed to reach the goal (ep {ep})")
                    history.append(episode_step)
                    break

                if random.random() < eps:
                    # do something random

                    if random.random() < action_option_sampling:
                        # follow random action
                        action = random.randint(0, self.env.action_space.n - 1)
                        state, _, _, _ = self.env.step(action)
                        episode_step += 1
                        self.update_sr(sr=new_sr, state=state, prev_state=prev_state)
                    else:
                        # follow random option
                        option = random.randint(0, self.options_count-1)
                        state, steps_used = self.follow_option_policy(prev_state, option_q=option_policies[option], option_subgoal_state=subgoal_states[option])
                        episode_step += steps_used
                        action = option+self.env.action_space.n
                else:
                    # greedy actions
                    if action_option_sampling == 1:
                        # only use primary actions
                        action = np.random.choice(np.flatnonzero(q[prev_state][0:self.env.action_space.n-1] == q[prev_state][0:self.env.action_space.n-1].max()))
                    else:
                        # use both action and options
                        action = np.random.choice(np.flatnonzero(q[prev_state] == q[prev_state].max()))

                    if action >= self.options_count:
                        # use option
                        option = action-self.env.action_space.n
                        state, steps_used = self.follow_option_policy(prev_state, option_q=option_policies[option], option_subgoal_state=subgoal_states[option], update_sr=new_sr)
                        episode_step += steps_used
                    else:
                        # use primary action
                        state, _, _, _ = self.env.step(action)
                        self.update_sr(sr=new_sr, state=state, prev_state=prev_state)
                        episode_step += 1

                # TD-update
                if state == goal_state:
                    reward = 1
                else:
                    reward = 0

                self.smdp_td_update(q, state, prev_state, action, reward)

                if state == goal_state:
                    self._log(f"found the goal state in {episode_step} steps")

                    cum_reward += 1
                    history.append(episode_step)
                    break

                prev_state = state

        self.viz.visualize_policy_learning_curve(history)

        return q, new_sr, history

    def follow_option_policy(self, state, option_q, option_subgoal_state, update_sr=None):

        eps = 0.1
        steps_used = 0
        prev_state = state

        while True:
            if random.random() < eps:
                # also sometimes take random action while following the option policy
                action = random.randint(0, self.env.action_space.n-1)
            else:
                action = np.random.choice(np.flatnonzero(option_q[state] == option_q[state].max()))

            state, _, _, _ = self.env.step(action=action)
            steps_used += 1

            if update_sr is not None:
                # update the SR for the incremental updating
                self.update_sr(update_sr, state, prev_state)

            if state == option_subgoal_state:
                # state is the end state the option ended in, steps_used is the amount of steps it took to get there
                return state, steps_used

            prev_state = state

    # smdp-runs, how many times do we learn/run the smdp (starting with developed options, but with a clean SMDP q-table)
    def run(self, iterations, action_option_sampling, smdp_runs=1):

        self.viz.visualize_env()

        initial_sr_filename = "initial_sr.npy"

        # first time get some random samples from the environment
        # if we did this before, re-use the SR
        sr = self._load_array(initial_sr_filename)

        if sr is None:
            # run the policy (completely random) for the first time
            sr = self.run_random_policy(steps=self.rollout_samples)

            # save the result, so next time no need to do this again
            self._save_array(sr, initial_sr_filename)

        self.viz.visualize_sr(sr)

        # the idea is that the subgoals should be better each iteration
        # TODO: meassure how many states can be reached from the subgoals
        # TODO: distance between subgoals
        for iteration in range(iterations):

            self._log(f"=== Running iteration {iteration} ===")

            # do the clustering, to find the subgoals
            sr_clusters = self.cluster_sr(sr)

            # filter possible subgoal states
            canidate_subgoals = self.get_candidate_subgoals(sr)

            # make a visual of all states that are valid subgoals
            self.viz.visualize_candidate_subgoals(canidate_subgoals, iteration)

            subgoal_states = self.get_subgoal_states(sr, sr_clusters, canidate_subgoals)

            # visualize each of the subgoals SR
            for subgoal_state in subgoal_states:
                state_sr = sr[subgoal_state]
                self.viz.visualize_sr_state(state_sr, subgoal_state, iteration)
                self.viz.visualize_subgoal_reward_map(state_sr, subgoal_state, iteration)

            self.viz.visualize_subgoals(subgoal_states, iteration=iteration)

            option_policies_filename = f"option_policies_{iteration}.npy"
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
                self.viz.visualize_policy(option_policies[option_id], None, subgoal_state, action_meaning=self.env._action_meaning, id="Option "+str(option_id+1))

            # run the SMDP
            all_smdp_history = []

            for run in range(smdp_runs):

                self._log(f"- SMDP RUN {run}")

                # find a goal/start position that is not a wall
                start_state = self.env.get_free_rand_state()
                end_goal = self.env.get_free_rand_state()

                smdp_q, sr, smdp_history = self.run_smdp(option_policies=option_policies, goal_state=end_goal, start_state=start_state, subgoal_states=subgoal_states, episodes=200, action_option_sampling=action_option_sampling)
                all_smdp_history.append(smdp_history)

                # vizualize the learned policy
                action_meaning_smdp = self.get_action_meaning_smdp()
                self.viz.visualize_policy(smdp_q, start_state, end_goal, action_meaning=action_meaning_smdp, id=f"SMDP_{iteration}_{run}")

            # TODO: fix this in the iterative approach
            avg_curve = np.average(np.array(all_smdp_history), axis=0)
            return avg_curve


# cli arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--gamma', default=0.95)
parser.add_argument('--seed', default=42)
parser.add_argument('--reset', default=False)
parser.add_argument('--options_count', default=4)
parser.add_argument('--iterations', default=1)
parser.add_argument('--ao_sampling', default=0.95) #  1 = more actions, 0 = more options
parser.add_argument('--env', default="FourRoom-v0")
parser.add_argument('--rollout_samples', default=int(5e6))
parser.add_argument('--option_learning_steps', default=int(1e6))


args = parser.parse_args()

DATA_DIR = os.path.join("data", f"{args.env}_oc{args.options_count}_alpha{args.alpha}_oa{args.ao_sampling}")

# set the random seed
random.seed(args.seed)

# create data dir if it does not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if args.reset:
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

so = SuccessorOptionsAgent(env_name=args.env,
                           alpha=args.alpha,
                           gamma=args.gamma,
                           rollout_samples=args.rollout_samples,
                           options_count=args.options_count,
                           option_learning_steps=args.option_learning_steps)

smdp_runs = 500

learning_history = so.run(iterations=int(args.iterations), smdp_runs=smdp_runs, action_option_sampling=float(args.ao_sampling))

np.save(os.path.join(DATA_DIR, "learning_history.npy"), np.array(learning_history))

#so_res = so.run(iterations=int(args.iterations), smdp_runs=smdp_runs, action_option_sampling=float(args.ao_sampling))
#so_un = so.run(iterations=int(args.iterations), smdp_runs=smdp_runs, action_option_sampling=0.5)

#plt.figure(figsize=(8, 8))
#plt.plot(so_un, label='SO uniform')
#plt.plot(so_res, label='SO')
#plt.plot(plain_q, label='Q')
#plt.ylim(0, 1000)

#plt.legend(loc='upper left')
#plt.xlabel("Episode")
#plt.ylabel("Steps required to reach goal")
#plt.savefig("compared.png")

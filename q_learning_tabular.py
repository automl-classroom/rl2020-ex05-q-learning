import gym
import numpy as np
import random
from collections import defaultdict

MAX_EPISODE_LENGTH = 250
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 1

env = gym.make('CartPole-v0')

BINS = 20
NUM_STATES = BINS ** 4

CART_POSITION = np.linspace(-4.8, 4.8, BINS)
CART_VELOCITY = np.linspace(-1, 1, BINS)
POLE_ANGLE = np.linspace(-0.418, 0.418, BINS)
POLE_ANGULAR_VELOCITY = np.linspace(-3, 3, BINS)


Q = defaultdict(lambda: np.random.uniform(1, -1))


def to_bins(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def to_discrete_state(obs):
    x, v, theta, omega = obs
    state = (to_bins(x, CART_POSITION),
             to_bins(v, CART_VELOCITY),
             to_bins(theta, POLE_ANGLE),
             to_bins(omega, POLE_ANGULAR_VELOCITY))
    return state


def policy(state, exploration_rate):
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample()
    q_values = [Q[(state, action)] for action in range(env.action_space.n)]
    return np.argmax(q_values)


def q_learning(num_episodes, exploration_rate=0.5, exploration_rate_decay=0.9, min_exploration_rate=0.01):
    rewards = []
    print("Performing Q-learning with %d states" % NUM_STATES)
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()
        state = to_discrete_state(obs)

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(state, exploration_rate)

            obs, reward, done, _ = env.step(action)

            next_state = to_discrete_state(obs)
            optimal_next_action = policy(next_state, exploration_rate)

            """ Implement Q-Learning Update"""

            state = next_state

            rewards[-1] += reward
            if done:
                break

        exploration_rate = max(exploration_rate_decay *
                               exploration_rate, min_exploration_rate)
        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100):]))
    return rewards


if __name__ == "__main__":
    q_learning(100000)

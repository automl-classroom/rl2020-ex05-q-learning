import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_EPISODE_LENGTH = 250
LEARNING_RATE = 1
DISCOUNT_FACTOR = 1

env = gym.make('CartPole-v0')


Q = nn.Linear(np.prod(env.observation_space.shape),
              env.action_space.n, bias=False)
Q.weight.data.uniform_(-0.0001, 0.0001)

optimizer = optim.SGD(Q.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = 64


def policy(state, exploration_rate):
    if np.random.uniform(0, 1) < exploration_rate:
        return env.action_space.sample() 
    q_values = Q(torch.from_numpy(state).float()).detach().numpy()
    return np.argmax(q_values)


def vfa_update(states, actions, rewards, dones, next_states):
    """Implement Value Function Training Step"""


def q_learning(num_episodes, exploration_rate=0.1):
    rewards = []
    vfa_update_data = []
    for episode in range(num_episodes):
        rewards.append(0)
        obs = env.reset()
        state = obs

        for t in range(MAX_EPISODE_LENGTH):
            action = policy(state, exploration_rate)

            obs, reward, done, _ = env.step(action)

            next_state = obs
            vfa_update_data.append((state, action, reward, done, next_state))

            state = next_state

            rewards[-1] += reward

            if len(vfa_update_data) >= BATCH_SIZE:
                vfa_update(*zip(*vfa_update_data))
                vfa_update_data.clear()

            if done:
                break

        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100):]))
    return rewards


if __name__ == "__main__":
    q_learning(10000)

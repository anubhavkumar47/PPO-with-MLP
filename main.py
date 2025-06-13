import csv
import math

import numpy as np
import torch
from project_files.enviroment import  Environment
from project_files.buffer import ReplayBuffer
from project_files.ppo import PPO



def train():
    env = Environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim)
    memory = ReplayBuffer(state_dim, action_dim)

    max_episodes = 3000
    max_time_steps = 150
    log_data = []
    batch_size = 512
    max_action=1
    epsilon_start = 1  # start fully random
    epsilon_end = 0  # end mostly greedy
    epsilon_decay = max_episodes  # linear decay over all episodes
    policy_noise = 0.05  # small Gaussian noise on policy actions
    noise_clip = 0.1
    total_actor_loss = 0
    total_critic_loss=0


    for episode in range(max_episodes):
        state = env.reset()
        total_reward, total_energy, total_aoi = 0, 0, 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * episode / 30)

        for t in range(max_time_steps):
            # ε-greedy action:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                # print("Action from En ",action)
                log_prob=0
            else:
                action, log_prob = ppo.select_action(state)
                # print("Action from Po ",action)
                # optional smoothing noise around the greedy action
                noise = np.random.normal(0, policy_noise, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            next_state, reward, done, energy, aoi = env.step(action)

            memory.add(state, action, reward, next_state, float(done),log_prob)
            state = next_state

            total_reward += reward
            total_energy += energy
            total_aoi += aoi

            if done:
                break
            if len(memory) >= batch_size * 5:
                actor_loss, critic_loss = ppo.update(memory)
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                memory.ptr = 0
                memory.size = 0

        log_data.append([episode + 1, total_reward/max_time_steps, total_aoi/max_time_steps, total_energy/max_time_steps, total_actor_loss/max_time_steps, total_critic_loss/max_time_steps])
        print(f"Episode {episode + 1}: Reward={total_reward/max_time_steps:.2f}, AoI={total_aoi/max_time_steps:.2f}, Energy={total_energy/max_time_steps:.2f}, Actor Loss={total_actor_loss/max_time_steps:.4f}, Critic Loss={total_critic_loss/max_time_steps:.4f}")

    with open("training_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward", "AoI", "Energy", "ActorLoss", "CriticLoss"])
        writer.writerows(log_data)


if __name__ == "__main__":
    train()
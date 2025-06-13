import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import ActorNetwork, CriticNetwork
from ActorNetwork import TransformerActor


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, update_epochs=10,batch_size =128):
        self.actor = TransformerActor(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_old = TransformerActor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.update_epochs = update_epochs
        self.batch_size =batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_old.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean = self.actor(state)
        cov_mat = torch.diag(torch.full((mean.size(-1),), 0.5)).to(self.device)
        dist = MultivariateNormal(mean, cov_mat)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy()[0], action_log_prob.detach()

    def evaluate(self, state, action):
        mean = self.actor(state)
        cov_mat = torch.diag_embed(torch.full(mean.shape, 0.5).to(self.device))
        dist = MultivariateNormal(mean, cov_mat)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state)
        return action_log_probs, torch.squeeze(state_values), entropy

    def update(self, memory):
        # Compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.reward), reversed(memory.not_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert memory to tensors safely
        old_states = torch.tensor(memory.state, dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(memory.action, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(memory.log_prob, dtype=torch.float32).to(self.device)

        actor_losses, critic_losses = [], []

        # PPO update with mini-batching
        for _ in range(self.update_epochs):
            for i in range(0, len(old_states), self.batch_size):
                batch_states = old_states[i:i + self.batch_size]
                batch_actions = old_actions[i:i + self.batch_size]
                batch_old_log_probs = old_log_probs[i:i + self.batch_size]
                batch_rewards = rewards[i:i + self.batch_size]

                log_probs, state_values, dist_entropy = self.evaluate(batch_states, batch_actions)

                # Debug shapes (optional)
                # print("log_probs:", log_probs.shape, "old_log_probs:", batch_old_log_probs.shape)

                ratios = torch.exp(log_probs - batch_old_log_probs)
                advantages = batch_rewards - state_values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = nn.MSELoss()(state_values, batch_rewards)

                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()

                actor_losses.append(loss_actor.item())
                critic_losses.append(loss_critic.item())

        # Update old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        return np.mean(actor_losses), np.mean(critic_losses)


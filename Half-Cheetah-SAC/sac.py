import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import random
from collections import deque

is_load = True
batch_size = 256
min_buffer_size = 1000
num_episodes = 40
max_timesteps = 1000
gamma = 0.99
tau = 0.005

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.array, zip(*batch))
        return state, action, next_state, reward, 1 - done

    def size(self):
        return len(self.buffer)

# Actor ağı
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # NaN sorununu önlemek için sınır koyduk
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action) * self.max_action
        return action, log_prob

# Critic ağı
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.l1(torch.cat([state, action], 1)))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# SAC Ajan
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.critic1 = Critic(state_dim, action_dim).cuda()
        self.critic2 = Critic(state_dim, action_dim).cuda()
        self.critic1_target = Critic(state_dim, action_dim).cuda()
        self.critic2_target = Critic(state_dim, action_dim).cuda()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device="cuda")
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]

    def soft_update(self, target_net, source_net, tau):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, replay_buffer, batch_size, gamma, tau):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to("cuda")
        action = torch.FloatTensor(action).to("cuda")
        next_state = torch.FloatTensor(next_state).to("cuda")
        reward = torch.FloatTensor(reward).unsqueeze(1).to("cuda")
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to("cuda")

        with torch.no_grad():
            next_action, log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * log_prob
            target_q = reward + not_done * gamma * target_q
            target_q = target_q.detach().squeeze(1)  # Boyutu düzeltmek için ekledik

        current_q1 = self.critic1(state, action).squeeze(1)  # Boyutu düzeltmek için ekledik
        current_q2 = self.critic2(state, action).squeeze(1)  # Boyutu düzeltmek için ekledik

        critic1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = torch.nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        actor_loss = (self.log_alpha.exp() * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic1_target, self.critic1, tau)
        self.soft_update(self.critic2_target, self.critic2, tau)

# Modeli kaydetme fonksiyonu
def save_model(agent, filename="sac_agent.pth"):
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "log_alpha": agent.log_alpha
    }, filename)
    print(f"Model ağırlıkları '{filename}' dosyasına kaydedildi.")

# Modeli yükleme fonksiyonu
def load_model(agent, filename="sac_agent.pth"):
    checkpoint = torch.load(filename)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic1.load_state_dict(checkpoint["critic1"])
    agent.critic2.load_state_dict(checkpoint["critic2"])
    agent.log_alpha = checkpoint["log_alpha"]
    print(f"Model ağırlıkları '{filename}' dosyasından yüklendi.")

# Ortam ve ajanı oluştur
env = gym.make("HalfCheetah-v5", render_mode="human")
agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]))

# Eğitilmiş ağırlıkları yükle
if is_load:
    load_model(agent)

replay_buffer = ReplayBuffer()

# Eğitim döngüsüne başla
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        replay_buffer.add(state, action, next_state, reward, done)

        if replay_buffer.size() > min_buffer_size:
            agent.train(replay_buffer, batch_size, gamma, tau)
        
        state = next_state
        if done:
            break

    print(f"Episode {episode}, Reward: {episode_reward}")

# Eğitim tamamlandığında modeli kaydet
save_model(agent)
env.close()

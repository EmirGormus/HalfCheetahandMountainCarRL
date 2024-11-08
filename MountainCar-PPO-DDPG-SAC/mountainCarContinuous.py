import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Ortak Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
gamma = 0.99
tau = 0.005  # target network update rate for DDPG and SAC

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device))

# Ortak Ağlar
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))
        return self.max_action * torch.tanh(self.fc3(action))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# SAC Ajan
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.alpha = 0.2
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Critic Güncellemesi
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            q_target = rewards + (1 - dones) * gamma * torch.min(next_q1, next_q2)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss1 = nn.functional.mse_loss(q1, q_target)
        critic_loss2 = nn.functional.mse_loss(q2, q_target)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # Actor Güncellemesi
        new_actions = self.actor(states)
        q_new = torch.min(self.critic1(states, new_actions), self.critic2(states, new_actions))
        actor_loss = (self.alpha - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Hedef Ağ Güncellemesi
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

# DDPG Ajan
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.target_actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Hedef Q değerlerini hesaplama
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + (1 - dones) * gamma * self.target_critic(next_states, next_actions)

        # Critic kaybı
        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor kaybı
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Hedef ağ güncellemesi
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

# PPO Ajan
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def update(self, replay_buffer, batch_size, old_log_probs):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Log-probabiliteleri hesapla
        log_probs = self.actor(states)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Politika kaybı (Clipped Objective)
        advantage = rewards - self.critic(states, actions)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value kaybı
        value_loss = nn.functional.mse_loss(self.critic(states, actions), rewards)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

def train(agent, env, replay_buffer, agent_name, num_episodes=2000, batch_size=64, reward_threshold=-0.01):
    episode_rewards = []
    old_log_probs = torch.zeros(batch_size, 1).to(device)  # PPO için başlangıç değeri

    for episode in range(num_episodes):
        state, _ = env.reset()  # state ve info'yu ayırıyoruz
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)  # numpy array'e dönüştürme
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if len(replay_buffer.buffer) > batch_size:
                if agent_name == "PPO":
                    agent.update(replay_buffer, batch_size, old_log_probs)  # PPO için old_log_probs geçiriyoruz
                else:
                    agent.update(replay_buffer, batch_size)
            
            # Eğer episode tamamlanmışsa ya da zaman aşımına uğramışsa, döngüyü bitir
            if done or truncated:
                break
                
        episode_rewards.append(episode_reward)
        print(f"Agent: {agent_name} | Episode: {episode + 1} | Reward: {episode_reward}")
        
        # Eğer ödül eşiğine ulaşılmışsa eğitimi sonlandır
        if episode_reward >= reward_threshold:
            print(f"{agent_name} hedef ödül eşiğine ulaştı. Eğitim sonlandırılıyor.")
            break

    return episode_rewards





# Eğitim Başlatma
replay_buffer = ReplayBuffer()
sac_agent = SACAgent(state_dim, action_dim, max_action)
ddpg_agent = DDPGAgent(state_dim, action_dim, max_action)
ppo_agent = PPOAgent(state_dim, action_dim, max_action)

sac_rewards = train(sac_agent, env, replay_buffer, "SAC")
ddpg_rewards = train(ddpg_agent, env, replay_buffer, "DDPG")
ppo_rewards = train(ppo_agent, env, replay_buffer, "PPO")

# Sonuçları Görselleştir
plt.plot(sac_rewards, label='SAC')
plt.plot(ddpg_rewards, label='DDPG')
plt.plot(ppo_rewards, label='PPO')
plt.legend()
plt.show()

from ddpg import Agent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train():
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True)


    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[17], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=6)

    np.random.seed(0)
    score_history = []

    for i in range(100):
        done = False
        score = 0
        obs, _ = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))
        if i % 25 == 0:
            agent.save_models()

    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

def load_trained():
    env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, reset_noise_scale=0.1, exclude_current_positions_from_observation=True, render_mode="human")


    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[17], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=6)

    agent.load_models()

    np.random.seed(0)
    score_history = []

    for i in range(20):
        done = False
        score = 0
        obs, _ = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            score += reward
            obs = new_state

        score_history.append(score)
        print("episode", i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]))


    plt.plot(score_history)
    plt.title('Score History')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig("grafik.png")

load_trained()
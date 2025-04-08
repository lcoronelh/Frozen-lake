import gym
import numpy as np
from time import sleep
from IPython.display import clear_output
import random as rd

# Crear el entorno con render_mode adecuado
def run():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.8
    gamma = 0.95

    epsilon = 1
    epsilon_decay = 0.0001
    epsilon_min = 0.01
    rng = np.random.default_rng()

    num_episodes = 10000
    max_steps = 100

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            action =    env.action_space.sample() # Actions: 0=left, 1=down, 2=right, 3=up
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])  # Q-learning update rule

            state = new_state
        
        env.close()

if __name__ == "__main__":
    run()
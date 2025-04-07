import gym
import numpy as np
from time import sleep
from IPython.display import clear_output
import random as rd

# Crear el entorno con render_mode adecuado
def run():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

    state = env.reset()[0]
    terminated = False
    truncated = False

    while (not terminated and not truncated):
        action =    env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
    
    env.close()

if __name__ == "__main__":
    run()
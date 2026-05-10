

from env.grid_env import GridEnv

env = GridEnv()

obs, _ = env.reset()

print(obs["x"].shape)
print(obs["edge_index"].shape)

from env.grid_env import GridEnv

# Create environment
env = GridEnv()

# Reset environment
state = env.reset()
print("Initial state length:", len(state))

# Dummy action (no change to generators)
action = [0] * len(env.net.gen)

# Take one step
state, reward, done = env.step(action)

print("Reward:", reward)
print("Done:", done)
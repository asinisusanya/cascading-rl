
from stable_baselines3 import PPO

from env.grid_env import GridEnv
from training.gcn_extractor import GCNExtractor

import numpy as np


# ======================================================
# SELECT EXPERIMENT
# ======================================================

EXPERIMENT = "ppo"

# Options:
#
# "ppo"
# "ppo_or"
# "gnn_ppo"
# "gnn_ppo_or"


# ======================================================
# LOAD ENVIRONMENT + MODEL
# ======================================================

# -----------------------------
# PPO
# -----------------------------

if EXPERIMENT == "ppo":

    env = GridEnv(
        training=False,
        use_gnn=False,
        use_or=False
    )

    model = PPO.load(
        "saved_models/ppo"
    )


# -----------------------------
# PPO + OR
# -----------------------------

elif EXPERIMENT == "ppo_or":

    env = GridEnv(
        training=False,
        use_gnn=False,
        use_or=True
    )

    model = PPO.load(
        "saved_models/ppo_or"
    )


# -----------------------------
# GNN + PPO
# -----------------------------

elif EXPERIMENT == "gnn_ppo":

    env = GridEnv(
        training=False,
        use_gnn=True,
        use_or=False
    )

    model = PPO.load(
        "saved_models/gnn_ppo"
    )


# -----------------------------
# GNN + PPO + OR
# -----------------------------

elif EXPERIMENT == "gnn_ppo_or":

    env = GridEnv(
        training=False,
        use_gnn=True,
        use_or=True
    )

    model = PPO.load(
        "saved_models/gnn_ppo_or"
    )


# ======================================================
# RL EVALUATION
# ======================================================

print("\n=== RL AGENT ===")

rl_results = []

NUM_EPISODES = 50

MAX_STEPS = 500

for ep in range(NUM_EPISODES):

    print(f"\nEpisode {ep+1}")

    obs, _ = env.reset(
        seed=42 + ep
    )

    step_count = 0

    done = False

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, done, _, _ = env.step(action)

        step_count += 1

        if step_count >= MAX_STEPS:

            done = True

    rl_results.append(step_count)

    print(
        f"Survived steps: {step_count}"
    )


# ======================================================
# BASELINE EVALUATION
# ======================================================

print("\n=== BASELINE (NO CONTROL) ===")

baseline_results = []

for ep in range(NUM_EPISODES):

    print(f"\nEpisode {ep+1}")

    obs, _ = env.reset(
        seed=42 + ep
    )

    step_count = 0

    done = False

    while not done:

        action = np.zeros(
            len(env.net.gen)
        )

        obs, reward, done, _, _ = env.step(action)

        step_count += 1

        if step_count >= MAX_STEPS:

            done = True

    baseline_results.append(step_count)

    print(
        f"Baseline survived steps: {step_count}"
    )


# ======================================================
# FINAL STATISTICS
# ======================================================

print("\n==============================")
print("FINAL STATISTICS")
print("==============================")

print("\n--- RL AGENT ---")

print(
    f"Mean: {np.mean(rl_results):.2f}"
)

print(
    f"Std: {np.std(rl_results):.2f}"
)

print(
    f"Min: {np.min(rl_results)}"
)

print(
    f"Max: {np.max(rl_results)}"
)

print(
    f"Median: {np.median(rl_results)}"
)

print("\n--- BASELINE ---")

print(
    f"Mean: {np.mean(baseline_results):.2f}"
)

print(
    f"Std: {np.std(baseline_results):.2f}"
)

print(
    f"Min: {np.min(baseline_results)}"
)

print(
    f"Max: {np.max(baseline_results)}"
)

print(
    f"Median: {np.median(baseline_results)}"
)


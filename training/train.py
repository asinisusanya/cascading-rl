
from stable_baselines3 import PPO

from env.grid_env import GridEnv
from training.gcn_extractor import GCNExtractor


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
# PPO ONLY
# ======================================================

if EXPERIMENT == "ppo":

    env = GridEnv(
        training=True,
        use_gnn=False,
        use_or=False
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        device="cuda"
    )


# ======================================================
# PPO + OR
# ======================================================

elif EXPERIMENT == "ppo_or":

    env = GridEnv(
        training=True,
        use_gnn=False,
        use_or=True
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        device="cuda"
    )


# ======================================================
# GNN + PPO
# ======================================================

elif EXPERIMENT == "gnn_ppo":

    env = GridEnv(
        training=True,
        use_gnn=True,
        use_or=False
    )

    policy_kwargs = dict(
        features_extractor_class=GCNExtractor,
        features_extractor_kwargs=dict(
            features_dim=128
        )
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        device="cuda"
    )


# ======================================================
# GNN + PPO + OR
# ======================================================

elif EXPERIMENT == "gnn_ppo_or":

    env = GridEnv(
        training=True,
        use_gnn=True,
        use_or=True
    )

    policy_kwargs = dict(
        features_extractor_class=GCNExtractor,
        features_extractor_kwargs=dict(
            features_dim=128
        )
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        device="cuda"
    )


# ======================================================
# TRAIN
# ======================================================

print("\nTraining started...\n")

model.learn(total_timesteps=50000)

print("\nTraining finished!\n")


# ======================================================
# SAVE MODEL
# ======================================================

model.save(
    f"saved_models/{EXPERIMENT}"
)

print(f"\nSaved model: {EXPERIMENT}\n")


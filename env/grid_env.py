
import pandapower as pp
import pandapower.networks as pn

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box

import numpy as np
import cvxpy as cp


class GridEnv(gym.Env):

    def __init__(
        self,
        training=True,
        use_gnn=True,
        use_or=True
    ):

        super(GridEnv, self).__init__()

        self.training = training
        self.use_gnn = use_gnn
        self.use_or = use_or

        # ==================================================
        # LOAD IEEE 39 BUS SYSTEM
        # ==================================================

        self.net = pn.case39()

        # ==================================================
        # ACTION SPACE
        # ==================================================

        self.action_space = spaces.Box(
            low=-10,
            high=10,
            shape=(len(self.net.gen),),
            dtype=np.float32
        )

        # ==================================================
        # OBSERVATION SPACE
        # ==================================================

        n_lines = len(self.net.line)
        n_gen = len(self.net.gen)

        # -----------------------------
        # GNN OBSERVATION
        # -----------------------------

        if self.use_gnn:

            self.observation_space = Dict({

                "x": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.net.bus), 2),
                    dtype=np.float32
                ),

                "edge_index": Box(
                    low=0,
                    high=100,
                    shape=(2, 500),
                    dtype=np.int64
                )
            })

        # -----------------------------
        # NORMAL VECTOR OBSERVATION
        # -----------------------------

        else:

            obs_size = (
                n_lines +      # loading
                n_lines +      # status
                n_gen +        # generation
                3              # global features
            )

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )

    # ======================================================
    # LOAD STRESS
    # ======================================================

    def increase_load(self, factor=1.15):

        self.net.load["p_mw"] *= factor

    # ======================================================
    # NORMAL VECTOR STATE
    # ======================================================

    def get_state(self):

        loading = self.net.res_line[
            "loading_percent"
        ].fillna(0).values

        status = self.net.line[
            "in_service"
        ].astype(int).values

        gen = self.net.gen["p_mw"].values

        max_loading = np.array([loading.max()])

        total_load = np.array([
            self.net.load["p_mw"].sum()
        ])

        failed_lines = np.array([
            (~self.net.line["in_service"]).sum()
        ])

        state = np.concatenate([
            loading,
            status,
            gen,
            max_loading,
            total_load,
            failed_lines
        ])

        return state.astype(np.float32)

    # ======================================================
    # GRAPH OBSERVATION
    # ======================================================

    def get_graph_obs(self):

        n_bus = len(self.net.bus)

        node_features = []

        # -----------------------------
        # NODE FEATURES
        # -----------------------------

        for i in range(n_bus):

            load = 0
            gen = 0

            # Load at bus
            load_rows = self.net.load[
                self.net.load["bus"] == i
            ]

            if len(load_rows) > 0:
                load = load_rows["p_mw"].sum()

            # Generator at bus
            gen_rows = self.net.gen[
                self.net.gen["bus"] == i
            ]

            if len(gen_rows) > 0:
                gen = gen_rows["p_mw"].sum()

            # NORMALIZATION
            node_features.append([
                load / 1000.0,
                gen / 1000.0
            ])

        x = np.array(
            node_features,
            dtype=np.float32
        )

        # -----------------------------
        # EDGE INDEX
        # -----------------------------

        edges = []

        for _, line in self.net.line.iterrows():

            if line["in_service"]:

                from_bus = int(line["from_bus"])
                to_bus = int(line["to_bus"])

                edges.append([from_bus, to_bus])
                edges.append([to_bus, from_bus])

        # Padding
        max_edges = 500

        while len(edges) < max_edges:
            edges.append([0, 0])

        edges = edges[:max_edges]

        edge_index = np.array(
            edges
        ).T.astype(np.int64)

        return {
            "x": x,
            "edge_index": edge_index
        }

    # ======================================================
    # APPLY RL ACTION
    # ======================================================

    def apply_action(self, action):

        for i in range(len(self.net.gen)):

            self.net.gen.at[i, "p_mw"] += action[i] * 4

    # ======================================================
    # OR LAYER
    # ======================================================

    def optimize_action(self, action):

        n_gen = len(self.net.gen)

        corrected = cp.Variable(n_gen)

        rl_action = np.array(action)

        objective = cp.Minimize(
            cp.sum_squares(corrected - rl_action)
        )

        constraints = []

        # Ramp constraints
        ramp_limit = 5

        constraints += [
            corrected <= ramp_limit,
            corrected >= -ramp_limit
        ]

        # Generator constraints
        current_gen = self.net.gen["p_mw"].values

        gen_min = 0
        gen_max = 1000

        constraints += [
            current_gen + corrected >= gen_min,
            current_gen + corrected <= gen_max
        ]

        # Power balance
        constraints += [
            cp.sum(corrected) == 0
        ]

        # Solve
        problem = cp.Problem(
            objective,
            constraints
        )

        problem.solve()

        # Safety fallback
        if corrected.value is None:

            return np.clip(
                action,
                -5,
                5
            )

        return corrected.value.astype(np.float32)

    # ======================================================
    # REWARD FUNCTION
    # ======================================================

    def compute_reward(self, action):

        overload = (
            self.net.res_line["loading_percent"] > 100
        )

        failed = (
            ~self.net.line["in_service"]
        ).sum()

        served = self.net.res_load["p_mw"].sum()

        total = self.net.load["p_mw"].sum()

        load_loss = total - served

        reward = 0

        # penalties
        reward -= 10 * overload.sum()
        reward -= 3 * failed
        reward -= 1 * load_loss

        # action penalty
        reward -= 0.02 * np.sum(np.abs(action))

        # survival bonus
        reward += 0.5

        # time penalty
        reward -= 0.1

        # system health
        overload_ratio = (
            overload.sum() / len(self.net.line)
        )

        reward += 2 * (1 - overload_ratio)

        # collapse penalty
        if self.net.res_bus["vm_pu"].isna().any():

            reward -= 50

        # REWARD CLIPPING
        reward = np.clip(
            reward,
            -100,
            100
        )

        return reward

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        self.step_count += 1

        # Natural fluctuation
        if (
            self.training and
            self.step_count % 5 == 0
        ):

            self.net.load["p_mw"] *= np.random.uniform(
                0.998,
                1.002
            )

        # -----------------------------
        # OR LAYER
        # -----------------------------

        if self.use_or:

            action = self.optimize_action(action)

        # Apply action
        self.apply_action(action)

        # Progressive stress
        if self.step_count % 3 == 0:

            self.net.load["p_mw"] *= np.random.uniform(
                1.0,
                1.005
            )

        # Run power flow
        pp.rundcpp(self.net)

        # Cascading failures
        for idx, loading in self.net.res_line[
            "loading_percent"
        ].items():

            if loading > 100:

                prob = min(
                    1.0,
                    (loading - 100) / 100
                )

                if np.random.rand() < prob:

                    self.net.line.at[
                        idx,
                        "in_service"
                    ] = False

        # Reward
        reward = self.compute_reward(action)

        # Observation
        if self.use_gnn:

            state = self.get_graph_obs()

        else:

            state = self.get_state()

        # Collapse check
        done = False

        if self.net.res_bus["vm_pu"].isna().any():

            done = True

        return state, reward, done, False, {}

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        if seed is not None:

            np.random.seed(seed)

        self.step_count = 0

        while True:

            self.net = pn.case39()

            # Stress
            self.increase_load(1.1)

            # Initial disturbance
            if self.training:

                line_to_trip = 5

            else:

                line_to_trip = np.random.choice(
                    self.net.line.index[:20]
                )

            self.net.line.at[
                line_to_trip,
                "in_service"
            ] = False

            # Run PF
            pp.rundcpp(self.net)

            loading = self.net.res_line[
                "loading_percent"
            ]

            # Too easy
            if loading.max() < 70:
                continue

            # Already collapsed
            if self.net.res_bus[
                "vm_pu"
            ].isna().any():

                continue

            break

        # Return observation
        if self.use_gnn:

            return self.get_graph_obs(), {}

        else:

            return self.get_state(), {}


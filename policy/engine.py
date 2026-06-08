import statistics
from collections import deque
from policy.monitor import SystemMonitor
from policy.qtable import DoubleQTable
from policy.replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Reward hyper-parameters
# ---------------------------------------------------------------------------

SENSITIVITY_WEIGHTS = {
    "low":    {"security": 0.25, "latency": 0.45, "cpu": 0.30},
    "medium": {"security": 0.40, "latency": 0.36, "cpu": 0.24},
    "high":   {"security": 0.65, "latency": 0.21, "cpu": 0.14},
}

SECURITY_SCORES = {"classical": 0.4, "hybrid": 0.7, "post_quantum": 1.0}

LATENCY_CAP_MS = 300.0
CPU_CAP_PCT    = 80.0

# Dynamic bonus coefficient (β): bonus = β · (1 − L/LATENCY_CAP)
HIGH_SENSITIVITY_PQ_BONUS_BETA = 0.15   # max bonus ≈ 0.15 when latency ≈ 0

# Penalty for weak crypto on high-sensitivity requests (paper §VI item 3)
HIGH_SENSITIVITY_WEAK_PENALTY = 0.20

# Adaptive re-exploration
TRAILING_WINDOW      = 100
REEXPLORE_EPSILON    = 0.20
REEXPLORE_THRESHOLD  = 1.0      # standard deviations below mean


class PolicyEngine:
   

    SENSITIVITIES = ("low", "medium", "high")

    def __init__(self) -> None:
        self.monitor = SystemMonitor(window=20)

        # Per-sensitivity Double Q-tables (fix for per-sensitivity anomaly)
        self.qtables: dict[str, DoubleQTable] = {
            s: DoubleQTable(sensitivity_label=s, alpha=0.1, gamma=0.0, epsilon=0.3)
            for s in self.SENSITIVITIES
        }

        # Per-sensitivity replay buffers
        self.buffers: dict[str, ReplayBuffer] = {
            s: ReplayBuffer(maxlen=5000, batch_size=32, train_every=50)
            for s in self.SENSITIVITIES
        }

        # Per-sensitivity trailing rewards for adaptive re-exploration
        self._trailing: dict[str, deque] = {
            s: deque(maxlen=TRAILING_WINDOW) for s in self.SENSITIVITIES
        }

        # History for plotting (per sensitivity)
        self.reward_history:  dict[str, list] = {s: [] for s in self.SENSITIVITIES}
        self.epsilon_history: dict[str, list] = {s: [] for s in self.SENSITIVITIES}
        self.decision_log: list[dict] = []

        # State carried between select_algorithm() and record_request_outcome()
        self._pending: dict | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_algorithm(
        self, sensitivity: str = "medium", payload_size: int = 100
    ) -> dict:
        """
        Select a cryptographic algorithm for the current request.
        Uses the Q-table that belongs to this sensitivity level.
        """
        sens = sensitivity if sensitivity in self.SENSITIVITIES else "medium"
        qtable = self.qtables[sens]
        snap = self.monitor.snapshot()

        state_idx = qtable.get_state_idx(
            cpu=snap["cpu_percent"],
            latency=snap["avg_latency"],
            payload=payload_size,
            trend=snap["cpu_trend"],
        )
        action_idx = qtable.select_action(state_idx)
        algorithm  = qtable.ACTIONS[action_idx]

        reasoning = (
            f"[{sens.upper()}] Double-Q selected {algorithm} | "
            f"cpu={snap['cpu_percent']:.1f}% lat={snap['avg_latency']:.1f}ms "
            f"ε={qtable.epsilon:.4f}"
        )

        self._pending = {
            "snap":       snap,
            "state_idx":  state_idx,
            "action_idx": action_idx,
            "sensitivity": sens,
            "algorithm":  algorithm,
        }

        self.decision_log.append({
            "algorithm":   algorithm,
            "cpu":         snap["cpu_percent"],
            "sensitivity": sens,
        })

        return {
            "algorithm": algorithm,
            "score": float(
                qtable.combined_values(state_idx)[action_idx]
            ),
            "reasoning": reasoning,
            "state_idx":  state_idx,
        }

    def record_request_outcome(self, total_latency_ms: float) -> None:
        """
        Called by the FastAPI middleware after each request completes.
        Updates the appropriate per-sensitivity Q-table and replay buffer.
        """
        if self._pending is None:
            return

        p   = self._pending
        sens = p["sensitivity"]
        self._pending = None

        self.monitor.record_latency(total_latency_ms)

        reward = self._compute_reward(
            alg=p["algorithm"],
            latency_ms=total_latency_ms,
            cpu_pct=p["snap"]["cpu_percent"],
            sensitivity=sens,
        )

        self.reward_history[sens].append(reward)
        self.epsilon_history[sens].append(self.qtables[sens].epsilon)
        self._trailing[sens].append(reward)

        buf = self.buffers[sens]
        buf.push(p["state_idx"], p["action_idx"], reward, None)

        if buf.should_train():
            for s, a, r, _ in buf.sample():
                self.qtables[sens].update(s, a, r)

        self._maybe_reexplore(sens, reward)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        alg: str,
        latency_ms: float,
        cpu_pct: float,
        sensitivity: str,
    ) -> float:
        w   = SENSITIVITY_WEIGHTS.get(sensitivity, SENSITIVITY_WEIGHTS["medium"])
        sec = SECURITY_SCORES[alg]
        lat = min(latency_ms / LATENCY_CAP_MS, 1.0)
        cpu = min(cpu_pct    / CPU_CAP_PCT,    1.0)

        reward = w["security"] * sec - w["latency"] * lat - w["cpu"] * cpu

        if sensitivity == "high":
            if alg == "post_quantum":
                # Dynamic bonus: large when latency is low, shrinks under load
                bonus = HIGH_SENSITIVITY_PQ_BONUS_BETA * (1.0 - lat)
                reward += bonus
            else:
                # Penalty for choosing weak crypto on high-sensitivity data
                reward -= HIGH_SENSITIVITY_WEAK_PENALTY

        return reward

    # ------------------------------------------------------------------
    # Adaptive re-exploration
    # ------------------------------------------------------------------

    def _maybe_reexplore(self, sensitivity: str, latest_reward: float) -> None:
        trail = self._trailing[sensitivity]
        if len(trail) < TRAILING_WINDOW:
            return
        mean = statistics.mean(trail)
        std  = statistics.stdev(trail)
        if std > 0 and latest_reward < (mean - REEXPLORE_THRESHOLD * std):
            qt = self.qtables[sensitivity]
            qt.epsilon = max(qt.epsilon, REEXPLORE_EPSILON)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        counts: dict[str, int] = {"classical": 0, "hybrid": 0, "post_quantum": 0}
        for d in self.decision_log[-1000:]:
            counts[d["algorithm"]] += 1

        per_sens_rewards: dict[str, float] = {}
        for s in self.SENSITIVITIES:
            hist = self.reward_history[s]
            window = hist[-100:] if len(hist) >= 100 else hist
            per_sens_rewards[s] = round(sum(window) / max(len(window), 1), 4)

        return {
            "total_decisions": len(self.decision_log),
            "distribution":    counts,
            "per_sensitivity_avg_reward": per_sens_rewards,
            "epsilons": {s: round(qt.epsilon, 4) for s, qt in self.qtables.items()},
            "q_table_updates": {s: qt.update_count for s, qt in self.qtables.items()},
            "buffer_sizes":    {s: len(b) for s, b in self.buffers.items()},
            "avg_latency_ms":  round(self.monitor.get_avg_latency(), 2),
        }

    def get_history(self) -> dict:
        """Full history for paper plots (replaces /policy/history JSON)."""
        return {
            "rewards":   self.reward_history,
            "epsilons":  self.epsilon_history,
            "decisions": self.decision_log,
        }

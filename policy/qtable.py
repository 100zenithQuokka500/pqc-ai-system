import numpy as np
import logging

logger = logging.getLogger(__name__)


class DoubleQTable:
   
    ACTIONS = ["classical", "hybrid", "post_quantum"]
    N_ACTIONS = 3

    # State discretisation boundaries
    CPU_BINS     = [0, 25, 50, 75, 100]
    LATENCY_BINS = [0, 30, 80, 150, 300]
    PAYLOAD_BINS = [0, 100, 500, 2000, 99_999]
    TREND_BINS   = [-100, -10, 10, 100]

    # 4 × 4 × 4 × 3 = 192 states per sensitivity level
    STATE_SIZE = 4 * 4 * 4 * 3

    def __init__(
        self,
        sensitivity_label: str,
        alpha: float = 0.1,
        gamma: float = 0.0,
        epsilon: float = 0.3,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.999,
    ) -> None:
        self.sensitivity_label = sensitivity_label
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.update_count = 0

        rng = np.random.default_rng()
        init = dict(low=-0.01, high=0.01, size=(self.STATE_SIZE, self.N_ACTIONS))
        self.q1: np.ndarray = rng.uniform(**init)
        self.q2: np.ndarray = rng.uniform(**init)

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def get_state_idx(
        self,
        cpu: float,
        latency: float,
        payload: int,
        trend: float,
    ) -> int:
        cpu_idx = int(min(np.digitize(cpu,     self.CPU_BINS[1:]),     3))
        lat_idx = int(min(np.digitize(latency, self.LATENCY_BINS[1:]), 3))
        pay_idx = int(min(np.digitize(payload, self.PAYLOAD_BINS[1:]), 3))
        trd_idx = int(min(np.digitize(trend,   self.TREND_BINS[1:]),   2))
        idx = (cpu_idx * 4 * 4 * 3) + (lat_idx * 4 * 3) + (pay_idx * 3) + trd_idx
        return int(min(idx, self.STATE_SIZE - 1))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state_idx: int) -> int:
       
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.N_ACTIONS))
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        combined = self.q1[state_idx] + self.q2[state_idx]
        return int(np.argmax(combined))

    # ------------------------------------------------------------------
    # Update (Double Q-Learning)
    # ------------------------------------------------------------------

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int | None = None,
    ) -> None:
       
        if np.random.random() < 0.5:
            current = self.q1[state_idx, action]
            self.q1[state_idx, action] = current + self.alpha * (reward - current)
        else:
            current = self.q2[state_idx, action]
            self.q2[state_idx, action] = current + self.alpha * (reward - current)
        self.update_count += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def combined_values(self, state_idx: int) -> np.ndarray:
       
        return self.q1[state_idx] + self.q2[state_idx]

    def best_action_label(self, state_idx: int) -> str:
        return self.ACTIONS[int(np.argmax(self.combined_values(state_idx)))]

    def __repr__(self) -> str:
        return (
            f"DoubleQTable(sensitivity={self.sensitivity_label!r}, "
            f"ε={self.epsilon:.4f}, updates={self.update_count})"
        )

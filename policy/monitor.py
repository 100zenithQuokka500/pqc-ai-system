import psutil, time
from collections import deque
import threading

class SystemMonitor:
   
    def __init__(self, window=20):
        self._latency_window = deque(maxlen=window)
        self._cpu_window = deque(maxlen=10)  # Last 10 CPU readings
        self._lock = threading.Lock()
        # Background thread for CPU polling
        self._running = True
        self._thread = threading.Thread(target=self._poll_cpu, daemon=True)
        self._thread.start()

    def _poll_cpu(self):
        """Background thread --- polls CPU every 500ms."""
        while self._running:
            cpu = psutil.cpu_percent(interval=0.5)
            with self._lock:
                self._cpu_window.append(cpu)

    def record_latency(self, ms: float):
        with self._lock:
            self._latency_window.append(ms)

    def get_cpu_percent(self) -> float:
        with self._lock:
            return self._cpu_window[-1] if self._cpu_window else 0.0

    def get_cpu_trend(self) -> float:
        """Positive = CPU rising, Negative = CPU falling. Range roughly -100 to +100."""
        with self._lock:
            if len(self._cpu_window) < 3:
                return 0.0
            recent = list(self._cpu_window)
            # Slope of last 5 readings
            half = len(recent) // 2
            return (sum(recent[half:]) / len(recent[half:]) -
                    sum(recent[:half]) / len(recent[:half]))

    def get_avg_latency(self) -> float:
        with self._lock:
            if not self._latency_window:
                return 0.0
            return sum(self._latency_window) / len(self._latency_window)

    def snapshot(self) -> dict:
        return {
            'cpu_percent': self.get_cpu_percent(),
            'cpu_trend': self.get_cpu_trend(),
            'avg_latency': self.get_avg_latency(),
            'memory_pct': psutil.virtual_memory().percent
        }

    def stop(self):
        self._running = False
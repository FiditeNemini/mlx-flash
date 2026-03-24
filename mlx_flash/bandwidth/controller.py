import time
import collections

class UnifiedBandwidthController:
    """
    PID-based control-theoretic bandwidth scheduler for Apple Silicon shared memory.
    """
    def __init__(self, target_degradation=0.05):
        self.target_degradation = target_degradation
        self.base_times = {}
        
        # PID Tuning (Asymmetric)
        self.Kp_brake = 5e8    # Aggressive cut: bytes/sec to cut per second of error
        self.Kp_accel = 5e7    # Gentle add: bytes/sec to add per second of spare time
        self.Ki = 1e7          # Integral gain
        
        self.integral = 0.0
        self.I_max = 5e8
        self.I_min = -5e8
        
        self.B_limit = 1e9     # Start optimistic: 1 GB/s
        self.B_max = 5e9       # Physical ceiling
        self.B_min = 1e7       # 10 MB/s absolute minimum to prevent starvation
        
        self.ema_alpha = 0.3
        self.current_ema = {}

        # Token Bucket State
        self.tokens = 2 * 1024 * 1024 # Start with 2MB capacity
        self.max_tokens = 2 * 1024 * 1024
        self.last_token_update = time.perf_counter()

    def update_stats(self, bytes_read: int, duration_sec: float):
        # Kept for interface compatibility with existing IO worker code
        pass

    def register_compute_time(self, layer_idx: int, t_comp: float):
        # 1. Warmup / Uncontended Baseline
        if layer_idx not in self.base_times:
            self.base_times[layer_idx] = t_comp
            self.current_ema[layer_idx] = t_comp
            return

        # Outlier rejection (e.g., OS context switch, SLC flush)
        if t_comp > self.base_times[layer_idx] * 3.0:
            return 

        # 2. Filter Update
        self.current_ema[layer_idx] = (self.ema_alpha * t_comp) + \
                                      ((1 - self.ema_alpha) * self.current_ema[layer_idx])

        # 3. Error Calculation
        t_target = self.base_times[layer_idx] * (1.0 + self.target_degradation)
        error = t_target - self.current_ema[layer_idx] 

        # 4. Deadband
        if abs(error) < (self.base_times[layer_idx] * 0.01):
            error = 0.0

        # 5. PID Math
        Kp = self.Kp_accel if error > 0 else self.Kp_brake
        p_term = Kp * error
        
        # Conditional Integration (Anti-windup)
        if not (self.B_limit >= self.B_max and error > 0) and \
           not (self.B_limit <= self.B_min and error < 0):
            self.integral += (self.Ki * error)
            self.integral = max(self.I_min, min(self.I_max, self.integral))

        # 6. Actuation Output
        self.B_limit = self.B_limit + p_term + self.integral
        self.B_limit = max(self.B_min, min(self.B_max, self.B_limit))

    def consume_tokens(self, requested_bytes: int) -> float:
        """Returns sleep time required to satisfy requested bytes."""
        now = time.perf_counter()
        elapsed = now - self.last_token_update
        self.last_token_update = now
        
        # Add new tokens based on B_limit
        new_tokens = elapsed * self.B_limit
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        
        if self.tokens >= requested_bytes:
            self.tokens -= requested_bytes
            return 0.0
            
        # Deficit handling
        deficit = requested_bytes - self.tokens
        self.tokens = 0.0 # Consume whatever we have
        sleep_sec = deficit / self.B_limit
        return sleep_sec

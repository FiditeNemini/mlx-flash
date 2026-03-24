import time
import collections

class UnifiedBandwidthController:
    """
    Manages Apple Silicon shared memory bandwidth by dynamically throttling 
    background IO when the GPU needs maximum memory bandwidth for compute.
    """
    def __init__(self, max_bandwidth_gb_s: float = 60.0):
        # Maximum theoretical unified memory bandwidth (e.g. M1 Max = 400 GB/s)
        # But we only care about the bandwidth available to the CPU/SSD path.
        self.max_bandwidth_bytes_s = max_bandwidth_gb_s * 1024**3
        
        # Track recent read speeds to estimate current system load
        self.recent_reads = collections.deque(maxlen=10) 
        
        # State toggle controlled by the GlobalScheduler
        self.gpu_is_busy = False
        
        # Throttle configuration
        self.throttle_sleep_sec = 0.001
        self.max_chunk_size_bytes = 16 * 1024 * 1024
        
    def set_gpu_state(self, is_busy: bool):
        """Called by the executor/scheduler right before and after mx.eval()"""
        self.gpu_is_busy = is_busy

    def update_stats(self, bytes_read: int, duration_sec: float):
        """Called by the IO worker after a pread()"""
        if duration_sec > 0:
            bw = bytes_read / duration_sec
            self.recent_reads.append(bw)

    def get_estimated_bandwidth(self) -> float:
        if not self.recent_reads:
            return self.max_bandwidth_bytes_s
        return sum(self.recent_reads) / len(self.recent_reads)

    def calculate_throttle(self, requested_chunk_size: int) -> tuple[int, float]:
        """
        Determines if the IO thread should read a smaller chunk, or sleep,
        based on the current GPU state and memory controller saturation.
        Returns: (approved_chunk_size, sleep_sec_before_read)
        """
        # 1. If GPU is idle, BLAST the IO (No throttling)
        if not self.gpu_is_busy:
            return min(requested_chunk_size, self.max_chunk_size_bytes), 0.0
            
        # 2. If GPU is busy, we must throttle. 
        # Check if the memory controller is showing signs of saturation.
        # Apple NVMe pread() latency spikes exponentially when the GPU is saturating
        # the UMA fabric.
        current_bw = self.get_estimated_bandwidth()
        
        # If bandwidth has dropped significantly below theoretical, the fabric is saturated.
        fabric_saturation_ratio = current_bw / self.max_bandwidth_bytes_s
        
        # Heuristic:
        # If saturation is high (ratio is low, e.g. < 0.2), we must aggressively back off
        # to let the GPU finish its MatMul.
        if fabric_saturation_ratio < 0.3:
            # Drop chunk size to 1MB to prevent locking the controller
            approved_chunk = min(requested_chunk_size, 1024 * 1024)
            # Sleep slightly to yield bus cycles
            sleep_sec = self.throttle_sleep_sec * 2 
            return approved_chunk, sleep_sec
            
        elif fabric_saturation_ratio < 0.6:
            # Drop chunk size to 4MB
            approved_chunk = min(requested_chunk_size, 4 * 1024 * 1024)
            return approved_chunk, self.throttle_sleep_sec
            
        else:
            # GPU is computing, but not saturating bandwidth (e.g. vector-matrix in decode)
            # Safe to use standard chunks, but maybe add a tiny sleep just in case.
            approved_chunk = min(requested_chunk_size, 8 * 1024 * 1024)
            return approved_chunk, 0.0

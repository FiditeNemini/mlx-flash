import time
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

class StreamingProfiler:
    """
    Passively collects timing metrics for IO and GPU compute across layers,
    and analyzes the data to identify bottlenecks.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StreamingProfiler, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.layer_stats = defaultdict(lambda: {'io_wait': 0.0, 'compute': 0.0, 'calls': 0})
        self.cache_stats = {'moe_hits': 0, 'moe_misses': 0, 'os_page_hits': 0, 'os_page_misses': 0}
        self.token_times = []
        self.start_time = time.perf_counter()
        
    def record_layer_pass(self, layer_idx: int, io_wait: float, compute: float):
        self.layer_stats[layer_idx]['io_wait'] += io_wait
        self.layer_stats[layer_idx]['compute'] += compute
        self.layer_stats[layer_idx]['calls'] += 1

    def record_token(self):
        self.token_times.append(time.perf_counter())

    def record_moe_cache(self, hit: bool):
        if hit: self.cache_stats['moe_hits'] += 1
        else: self.cache_stats['moe_misses'] += 1

    def record_pread(self, duration: float, bytes_read: int):
        """
        Heuristic: if a 16MB pread takes < 1ms, it was highly likely in the 
        macOS Unified Page Cache (Warm). If it takes longer, it hit the NVMe SSD (Cold).
        """
        # Threshold: > 2GB/s indicates it was already in RAM.
        # e.g., 16MB in < 8ms
        threshold_s = bytes_read / (2.0 * 1024 * 1024 * 1024) 
        if duration < threshold_s:
            self.cache_stats['os_page_hits'] += 1
        else:
            self.cache_stats['os_page_misses'] += 1

    def analyze_bottlenecks(self) -> str:
        """The 'Oracle' - identifies where the system is bottlenecked."""
        total_io = sum(s['io_wait'] for s in self.layer_stats.values())
        total_compute = sum(s['compute'] for s in self.layer_stats.values())
        
        durations = [self.token_times[i] - self.token_times[i-1] for i in range(1, len(self.token_times))]
        avg_tps = (len(durations) / sum(durations)) if durations and sum(durations) > 0 else 0
        avg_latency = (sum(durations) / len(durations)) if durations else 0
        
        total_moe = self.cache_stats['moe_hits'] + self.cache_stats['moe_misses']
        moe_miss_rate = self.cache_stats['moe_misses'] / max(1, total_moe)
        
        report = []
        report.append("\n" + "="*50)
        report.append("🔍 MLX-FLASH PROFILER ORACLE")
        report.append("="*50)
        
        report.append(f"Tokens/Sec : {avg_tps:.2f} tok/s")
        report.append(f"Avg Latency: {avg_latency*1000:.1f} ms/tok")
        report.append(f"Total IO Wait  : {total_io:.2f} s")
        report.append(f"Total Compute  : {total_compute:.2f} s")
        report.append(f"MoE Miss Rate  : {moe_miss_rate*100:.1f}%")
        
        report.append("\n--- DIAGNOSIS ---")
        if moe_miss_rate > 0.8 and avg_latency > 3.0:
            report.append("⚠️  STATE: THRASHING")
            report.append("The working set size exceeds available RAM. The OS is evicting actively needed pages.")
            report.append("Recommendation: Lower `expert_cache_size`, decrease KV cache max size, or use a smaller/higher-quantized model.")
        elif total_io > total_compute * 1.5:
            report.append("⚠️  STATE: IO-BOUND (GPU is Starving)")
            report.append("The SSD cannot supply weights fast enough. Pipelining is waiting on disk reads.")
            report.append("Recommendation: Reduce weight precision (e.g. Q4_0 -> Q3_K) or use a faster NVMe drive.")
        elif total_compute > total_io * 1.5:
            report.append("✅  STATE: COMPUTE-BOUND")
            report.append("The SSD is fetching weights faster than the GPU can multiply them. Pipelining is hiding the IO perfectly.")
            report.append("Recommendation: You have spare IO bandwidth. You can afford to increase weight precision (e.g. Q4_0 -> Q6_K) for better quality.")
        else:
            report.append("✅  STATE: BALANCED")
            report.append("IO and Compute are perfectly matched.")
            
        return "\n".join(report)

    def print_waterfall(self):
        """Prints a visual ASCII waterfall chart of Layer IO vs Compute."""
        if not self.layer_stats: return
        
        print("\n--- LAYER WATERFALL (Avg ms per call) ---")
        max_time = max((s['io_wait'] + s['compute']) / max(1, s['calls']) for s in self.layer_stats.values())
        
        # Terminal width budget for the bar chart
        bar_width = 40 
        
        for layer_idx in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_idx]
            calls = max(1, stats['calls'])
            avg_io_ms = (stats['io_wait'] / calls) * 1000
            avg_comp_ms = (stats['compute'] / calls) * 1000
            total_ms = avg_io_ms + avg_comp_ms
            
            io_chars = int((avg_io_ms / (max_time * 1000)) * bar_width) if max_time > 0 else 0
            comp_chars = int((avg_comp_ms / (max_time * 1000)) * bar_width) if max_time > 0 else 0
            
            bar = "\033[91m" + ("=" * io_chars) + "\033[0m" + "\033[92m" + ("=" * comp_chars) + "\033[0m"
            # Pad to keep alignment
            pad = " " * (bar_width - (io_chars + comp_chars))
            
            print(f"L{layer_idx:02d} | {bar}{pad} | {total_ms:5.1f}ms (IO: {avg_io_ms:4.1f}, Comp: {avg_comp_ms:4.1f})")

    def export(self, filepath="/tmp/mlx_flash_profile.json"):
        report = {
            "layer_stats": dict(self.layer_stats),
            "cache_stats": self.cache_stats,
            "token_times": self.token_times
        }
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

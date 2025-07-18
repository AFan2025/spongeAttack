import threading
import time
import matplotlib.pyplot as plt
from pynvml import *
import numpy as np

class MultiGPUMonitor:
    def __init__(self, interval=0.5):
        nvmlInit()
        self.gpu_count = nvmlDeviceGetCount()
        self.interval = interval
        self.running = False

        #stats format is a dict of all of the gpus with a dict of each stat that holds numpy arrays
        self.stats = {i: {"timestamp": [],
                           "gpu_util": [],
                           "mem_used_MB": [],
                           "power_W": []}
                       for i in range(self.gpu_count)}
        
    def get_stats(self):
        return {
            f"GPU_{i}": {
                "max_gpu_util": max(self.stats[i]["gpu_util"]),
                "max_mem_used_MB": max(self.stats[i]["mem_used_MB"]),
                "max_power_W": max(self.stats[i]["power_W"])
            } for i in range(self.gpu_count)
        }

    def _poll(self):
        while self.running:
            for i in range(self.gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000

                self.stats[i]["timestamp"].append(time.time())
                self.stats[i]["gpu_util"].append(util.gpu)
                self.stats[i]["mem_used_MB"].append(mem.used / 1024**2)
                self.stats[i]["power_W"].append(power)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def clear_stats(self):
        self.stats = {i: {"timestamp": [],
                    "gpu_util": [],
                    "mem_used_MB": [],
                    "power_W": []}
                for i in range(self.gpu_count)}

    def plot(self):
        fig, axs = plt.subplots(self.gpu_count, 1, figsize=(10, 5 * self.gpu_count))

        if self.gpu_count == 1:
            axs = [axs]

        for i in range(self.gpu_count):
            timestamps = [stat["timestamp"] - self.stats[i][0]["timestamp"] for stat in self.stats[i]]
            utils = [stat["gpu_util"] for stat in self.stats[i]]
            mems = [stat["mem_used_MB"] for stat in self.stats[i]]
            powers = [stat["power_W"] for stat in self.stats[i]]

            axs[i].plot(timestamps, utils, label="Utilization (%)")
            # axs[i].plot(timestamps, mems, label="Memory (MB)")
            # axs[i].plot(timestamps, powers, label="Power (W)")
            axs[i].set_title(f"GPU {i}")
            axs[i].set_xlabel("Time (s)")
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

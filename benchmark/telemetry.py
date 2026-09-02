"""Background system telemetry sampler for benchmark runs.

One ``TelemetrySampler`` thread samples, at a fixed interval (default 0.5 s):

* GPU via NVML: utilization, memory-controller utilization, VRAM, power, SM/mem clocks,
  temperature, PCIe RX/TX throughput, performance state and clock-event (throttle) reasons;
* CPU via ``/proc/stat``: total and per-core busy fraction, plus mean core frequency from
  cpufreq;
* RAM via ``/proc/meminfo`` and page faults via ``/proc/vmstat``;
* optionally the server's ``/v1/stats`` (cheap; excluded from the server's access log).

Samples stream to a JSONL file; ``stop()`` returns a summary dict (means/maxima, integrated
PCIe bytes, throttle incidents, page-fault deltas). ``mark()`` records phase boundaries
(e.g. request start / first token / done) into the same JSONL so plots can split phases.

Memory-bandwidth counters and RAPL package power need root on this box (perf_event_paranoid=4,
energy_uj mode 0400) and are reported as unavailable rather than approximated.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any

try:
    import pynvml
except Exception:  # pragma: no cover - NVML is present on the bench box
    pynvml = None

_NCPU = os.cpu_count() or 1


def _read_proc_stat() -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """(busy, total) jiffies for the whole machine and per core."""
    total = (0, 0)
    cores: list[tuple[int, int]] = []
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            parts = line.split()
            vals = [int(x) for x in parts[1:11]]
            idle = vals[3] + vals[4]  # idle + iowait
            tot = sum(vals)
            if parts[0] == "cpu":
                total = (tot - idle, tot)
            else:
                cores.append((tot - idle, tot))
    return total, cores


def _read_meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "MemFree", "Cached", "Buffers"):
                out[k] = int(v.split()[0]) * 1024
    return out


def _read_vmstat() -> dict[str, int]:
    out = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, _, v = line.partition(" ")
            if k in ("pgfault", "pgmajfault"):
                out[k] = int(v)
    return out


def _read_cpufreq_khz() -> float | None:
    """Mean scaling_cur_freq across cores (kHz); None when unsupported."""
    tot = n = 0
    for cpu in range(0, _NCPU, 4):  # every 4th core is plenty for a mean
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
                tot += int(f.read())
                n += 1
        except OSError:
            return None
    return tot / n if n else None


class TelemetrySampler:
    def __init__(self, path: str, interval_s: float = 0.5, stats_url: str | None = None,
                 stats_every: int = 4) -> None:
        self.path = path
        self.interval_s = interval_s
        self.stats_url = stats_url
        self.stats_every = stats_every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._marks: list[dict[str, Any]] = []
        self._samples: list[dict[str, Any]] = []
        self._nv = None
        self._err: str | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "TelemetrySampler":
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nv = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:  # pragma: no cover
                self._err = f"nvml: {e}"
        self._t0 = time.monotonic()
        self._cpu0, self._cores0 = _read_proc_stat()
        self._vm0 = _read_vmstat()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="telemetry")
        self._thread.start()
        return self

    def mark(self, label: str) -> None:
        self._marks.append({"kind": "mark", "label": label, "t": time.monotonic() - self._t0,
                            "wall": time.time()})

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4 + 2)
        with open(self.path, "w") as f:
            for rec in self._marks + self._samples:
                f.write(json.dumps(rec) + "\n")
        summary = self._summarize()
        if pynvml is not None and self._nv is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return summary

    # ---------------------------------------------------------------- sampling
    def _sample_nvml(self, rec: dict[str, Any]) -> None:
        h = self._nv
        if h is None:
            return
        N = pynvml
        try:
            u = N.nvmlDeviceGetUtilizationRates(h)
            rec["gpu_util"] = u.gpu
            rec["gpu_mem_ctrl_util"] = u.memory
            rec["gpu_power_w"] = N.nvmlDeviceGetPowerUsage(h) / 1000.0
            rec["gpu_clock_sm_mhz"] = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
            rec["gpu_clock_mem_mhz"] = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)
            rec["gpu_temp_c"] = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
            rec["gpu_mem_used_mib"] = N.nvmlDeviceGetMemoryInfo(h).used >> 20
            rec["gpu_pstate"] = N.nvmlDeviceGetPerformanceState(h)
        except Exception:
            pass
        try:
            rec["gpu_throttle"] = N.nvmlDeviceGetCurrentClocksEventReasons(h)
        except Exception:
            try:
                rec["gpu_throttle"] = N.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            except Exception:
                pass
        try:
            # NVML samples PCIe counters over ~20 ms internally; KB/s.
            rec["pcie_tx_kbs"] = N.nvmlDeviceGetPcieThroughput(h, N.NVML_PCIE_UTIL_TX_BYTES)
            rec["pcie_rx_kbs"] = N.nvmlDeviceGetPcieThroughput(h, N.NVML_PCIE_UTIL_RX_BYTES)
        except Exception:
            pass

    def _loop(self) -> None:
        prev_total, prev_cores = self._cpu0, self._cores0
        core_busy_acc = [0.0] * len(prev_cores)
        n_cpu_samples = 0
        k = 0
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            rec: dict[str, Any] = {"kind": "sample", "t": round(t, 3)}
            self._sample_nvml(rec)
            total, cores = _read_proc_stat()
            db, dt = total[0] - prev_total[0], total[1] - prev_total[1]
            if dt > 0:
                rec["cpu_util"] = round(db / dt, 4)
                for i, (c, p) in enumerate(zip(cores, prev_cores)):
                    cdt = c[1] - p[1]
                    if cdt > 0:
                        core_busy_acc[i] += (c[0] - p[0]) / cdt
                n_cpu_samples += 1
            prev_total, prev_cores = total, cores
            mem = _read_meminfo()
            rec["ram_used_gib"] = round((mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)) / 2**30, 2)
            f = _read_cpufreq_khz()
            if f is not None:
                rec["cpu_freq_mhz"] = round(f / 1000.0)
            if self.stats_url and k % self.stats_every == 0:
                try:
                    with urllib.request.urlopen(self.stats_url, timeout=2) as r:
                        s = json.loads(r.read())
                    rec["srv_decode_tps"] = (s.get("throughput") or {}).get("decode_tps")
                    rec["srv_prefill_tps"] = (s.get("throughput") or {}).get("prefill_tps")
                    kv = s.get("kv") or {}
                    rec["srv_kv_used_pages"] = kv.get("used_pages")
                    rec["srv_kv_total_pages"] = kv.get("total_pages")
                    rec["srv_active"] = (s.get("requests") or {}).get("active")
                except Exception:
                    pass
            k += 1
            self._samples.append(rec)
            self._stop.wait(self.interval_s)
        self._core_busy_mean = (
            [round(a / n_cpu_samples, 3) for a in core_busy_acc] if n_cpu_samples else []
        )

    # ---------------------------------------------------------------- summary
    def _summarize(self) -> dict[str, Any]:
        s = self._samples
        vm1 = _read_vmstat()

        def col(key):
            return [r[key] for r in s if key in r and r[key] is not None]

        def mean(key):
            v = col(key)
            return round(sum(v) / len(v), 2) if v else None

        def mx(key):
            v = col(key)
            return max(v) if v else None

        dur = (s[-1]["t"] - s[0]["t"]) if len(s) > 1 else 0.0
        pcie_rx = col("pcie_rx_kbs")
        pcie_tx = col("pcie_tx_kbs")
        throttled = [r for r in s if r.get("gpu_throttle")]
        return {
            "samples": len(s),
            "duration_s": round(dur, 1),
            "nvml_error": self._err,
            "gpu_util_mean": mean("gpu_util"), "gpu_util_max": mx("gpu_util"),
            "gpu_mem_ctrl_util_mean": mean("gpu_mem_ctrl_util"),
            "gpu_power_w_mean": mean("gpu_power_w"), "gpu_power_w_max": mx("gpu_power_w"),
            "gpu_clock_sm_mhz_mean": mean("gpu_clock_sm_mhz"),
            "gpu_temp_c_max": mx("gpu_temp_c"),
            "gpu_mem_used_mib_max": mx("gpu_mem_used_mib"),
            "pcie_rx_gib_est": round(sum(pcie_rx) / len(pcie_rx) * dur / 2**20, 2) if pcie_rx and dur else None,
            "pcie_tx_gib_est": round(sum(pcie_tx) / len(pcie_tx) * dur / 2**20, 2) if pcie_tx and dur else None,
            "pcie_rx_kbs_mean": mean("pcie_rx_kbs"), "pcie_tx_kbs_mean": mean("pcie_tx_kbs"),
            "cpu_util_mean": mean("cpu_util"), "cpu_util_max": mx("cpu_util"),
            "cpu_freq_mhz_mean": mean("cpu_freq_mhz"),
            "cpu_core_busy_mean": getattr(self, "_core_busy_mean", []),
            "ram_used_gib_max": mx("ram_used_gib"),
            "pgfault_delta": vm1.get("pgfault", 0) - self._vm0.get("pgfault", 0),
            "pgmajfault_delta": vm1.get("pgmajfault", 0) - self._vm0.get("pgmajfault", 0),
            "throttle_samples": len(throttled),
            "throttle_reasons": sorted({hex(r["gpu_throttle"]) for r in throttled}),
            "mem_bandwidth": "unavailable (perf_event_paranoid=4, no root)",
            "cpu_pkg_power": "unavailable (RAPL energy_uj is root-only)",
        }

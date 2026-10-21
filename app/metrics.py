"""Minimal dependency-free Prometheus metrics (text exposition format 0.0.4).

Thread-safe. Run gunicorn with ONE worker (plus threads) so a single registry sees all requests;
scale horizontally with Kubernetes replicas.
"""
import threading
import time

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _esc(v):
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt(labels):
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in labels) + "}" if labels else ""


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._help, self._type = {}, {}
        self._counters, self._gauges, self._hists, self._buckets = {}, {}, {}, {}
        self.started = time.time()

    @staticmethod
    def _key(labels):
        return tuple(sorted((labels or {}).items()))

    def inc(self, name, labels=None, value=1.0, help=""):
        with self._lock:
            self._help.setdefault(name, help)
            self._type[name] = "counter"
            s = self._counters.setdefault(name, {})
            k = self._key(labels)
            s[k] = s.get(k, 0.0) + value

    def set_gauge(self, name, value, labels=None, help=""):
        with self._lock:
            self._help.setdefault(name, help)
            self._type[name] = "gauge"
            self._gauges.setdefault(name, {})[self._key(labels)] = float(value)

    def observe(self, name, value, labels=None, buckets=DEFAULT_BUCKETS, help=""):
        with self._lock:
            self._help.setdefault(name, help)
            self._type[name] = "histogram"
            self._buckets[name] = tuple(buckets)
            s = self._hists.setdefault(name, {})
            k = self._key(labels)
            if k not in s:
                s[k] = [[0] * len(buckets), 0.0, 0]
            for i, b in enumerate(buckets):
                if value <= b:
                    s[k][0][i] += 1
            s[k][1] += value
            s[k][2] += 1

    def render(self):
        lines = []
        with self._lock:
            self._gauges.setdefault("process_uptime_seconds", {})[()] = time.time() - self.started
            self._help.setdefault("process_uptime_seconds", "Seconds since the API process started")
            self._type["process_uptime_seconds"] = "gauge"
            for name in sorted(self._type):
                kind = self._type[name]
                lines += [f"# HELP {name} {self._help.get(name, '')}", f"# TYPE {name} {kind}"]
                if kind in ("counter", "gauge"):
                    store = self._counters if kind == "counter" else self._gauges
                    for labels, val in sorted(store.get(name, {}).items()):
                        lines.append(f"{name}{_fmt(labels)} {val:g}")
                else:
                    buckets = self._buckets[name]
                    for labels, (counts, total, n) in sorted(self._hists.get(name, {}).items()):
                        for b, c in zip(buckets, counts):
                            lines.append(f"{name}_bucket{_fmt(labels + (('le', f'{b:g}'),))} {c}")
                        lines.append(f"{name}_bucket{_fmt(labels + (('le', '+Inf'),))} {n}")
                        lines.append(f"{name}_sum{_fmt(labels)} {total:.6f}")
                        lines.append(f"{name}_count{_fmt(labels)} {n}")
        return "\n".join(lines) + "\n"

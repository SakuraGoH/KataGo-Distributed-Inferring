"""End-to-end verification of KataGo distributed inference across two rented GPU nodes.

Topology (exactly the production shape):

    local katago client (build-dist)
      -> 127.0.0.1:<localA> -> SakuraLink SSH tunnel -> weste:17731 -> katago distserver (RTX 4080S)
      -> 127.0.0.1:<localB> -> SakuraLink SSH tunnel -> westc:17731 -> katago distserver (RTX 4080S)

Runs: raw-NN byte equality vs the local CPU backend, per-node and combined
benchmarks, GPU utilisation sampling, and SSH RTT measurement.
"""
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, r"C:\Users\10632\Desktop\SakuraLink")
import sshutil  # noqa: E402
from sakura.config import NodeConfig  # noqa: E402
from sakura.tunnel import SSHTunnel  # noqa: E402

KATAGO_CLIENT = Path(r"C:\Users\10632\Desktop\KataGo-master\build-dist\Release\katago.exe")
KATAGO_LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\build-eigen\Release\katago.exe")
MODEL = Path(r"C:\Users\10632\Desktop\model.bin.gz")
OUT = CLOUD / "out"
REMOTE_DS_PORT = 17731

RAW_SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b all\nquit\n"


def make_cfg(path, servers, nn_threads, batch=8, extra=None):
    lines = [
        "logToStderr = false",
        "logAllGTPCommunication = false",
        "logSearchInfo = false",
        "rules = tromp-taylor",
        "allowResignation = false",
        "resignThreshold = -0.90",
        "resignConsecTurns = 3",
        "ponderingEnabled = false",
        "maxTimePondering = 60.0",
        "lagBuffer = 1.0",
        "maxVisits = 1",
        "numSearchThreads = 1",
        "searchFactorAfterOnePass = 0.50",
        "searchFactorAfterTwoPass = 0.25",
        "searchFactorWhenWinning = 0.40",
        "searchFactorWhenWinningThreshold = 0.95",
        "nnMaxBatchSize = %d" % batch,
        "numNNServerThreadsPerModel = %d" % nn_threads,
        "nnRandomize = false",
        "searchRandSeed = 12345",
    ]
    if servers:
        lines.append("distServers = %s" % servers)
        lines.append("distTimeoutSec = 600")
    for k, v in (extra or {}).items():
        lines.append("%s = %s" % (k, v))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def gtp(exe, cfg, script, timeout=300):
    r = subprocess.run([str(exe), "gtp", "-model", str(MODEL), "-config", str(cfg)],
                       input=script.encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    return r


def raw_text(stdout):
    out = []
    for ln in stdout.decode("utf-8", "replace").splitlines():
        if ln.startswith("= "):
            out.append(ln[2:])
        elif ln.startswith("="):
            out.append(ln[1:].lstrip())
        elif ln.startswith("?"):
            continue
        else:
            out.append(ln)
    return "\n".join(out).strip()


def bench(cfg, batch, iters, warmup=2, timeout=600):
    r = subprocess.run([str(KATAGO_CLIENT), "benchmarknn", "-model", str(MODEL),
                        "-config", str(cfg), "-batch-size", str(batch),
                        "-iterations", str(iters), "-warmup", str(warmup), "-json"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    lines = [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l.strip().startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError("benchmarknn failed: %s" % r.stderr.decode("utf-8", "replace")[-600:])
    d = json.loads(lines[-1])
    return {"evals_s": d["actualWallNNEvalsPerSec"],
            "ms_per_batch": d["perThreadMedianMs"][0] if d["perThreadMedianMs"] else 0,
            "threads": d["numThreads"], "batch": d["batchSize"]}


def ssh_rtt(client, n=20):
    """Round-trip time over the SSH control channel (keepalive global request)."""
    t = client.get_transport()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            t.global_request("keepalive@openssh.com", wait=True)
        except Exception:
            return None
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def start_distserver(name, gpus="0", model="/root/model.bin.gz"):
    c = sshutil.connect(name)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.3; true")
    cmd = ("cd /root/cpp && setsid nohup ./build-cuda/katago distserver "
           "-model %s -listen 127.0.0.1:%d -gpus %s "
           "-max-batch 256 -boardsize 19 -timeout 600 "
           "> /root/distserver.log 2>&1 < /dev/null &" % (model, REMOTE_DS_PORT, gpus))
    sshutil.run(c, cmd)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            return c
        time.sleep(0.5)
    rc, out = sshutil.run(c, "tail -n 20 /root/distserver.log")
    raise RuntimeError("%s distserver did not start:\n%s" % (name, out))


class GpuSampler(threading.Thread):
    def __init__(self, clients):
        super().__init__(daemon=True)
        self.clients = clients
        self.samples = {h: [] for h in clients}
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            for h, c in self.clients.items():
                try:
                    rc, out = sshutil.run(
                        c, "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                           "--format=csv,noheader,nounits", timeout=10)
                    if out.strip():
                        util, mem = out.strip().splitlines()[0].split(",")
                        self.samples[h].append((int(util), int(mem)))
                except Exception:
                    pass
            time.sleep(0.4)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=5)

    def summary(self):
        out = {}
        for h, vals in self.samples.items():
            if values := [v for v in vals if v is not None]:
                out[h] = {"max_util": max(v[0] for v in values),
                          "avg_util": round(sum(v[0] for v in values) / len(values), 1),
                          "max_mem_mb": max(v[1] for v in values)}
            else:
                out[h] = {"max_util": 0, "avg_util": 0, "max_mem_mb": 0}
        return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cloud_e2e_"))
    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}

    def step(name, ok, detail, **data):
        results["steps"].append({"step": name, "ok": bool(ok), "detail": detail, **data})
        print(("[PASS] " if ok else "[FAIL] ") + name + "  " + detail, flush=True)

    print("=== connect + start distservers ===", flush=True)
    clients = {}
    for h in ("weste", "westc"):
        clients[h] = start_distserver(h)
        print("  %s distserver up" % h, flush=True)
    step("both distservers started", True, "weste+westc, port %d" % REMOTE_DS_PORT)

    rtts = {h: ssh_rtt(c) for h, c in clients.items()}
    print("  SSH RTT (median): %s" % rtts, flush=True)
    results["ssh_rtt_ms"] = rtts

    tunnels = {}
    for h in ("weste", "westc"):
        n = NodeConfig(host=sshutil.HOSTS[h][0], username="root",
                       password=sshutil.HOSTS[h][2], remote_port=sshutil.HOSTS[h][1],
                       target_host="127.0.0.1", target_port=REMOTE_DS_PORT, name=h)
        t = SSHTunnel(n)
        t0 = time.perf_counter()
        t.start()
        tunnels[h] = t
        print("  tunnel %s: 127.0.0.1:%d (%.0f ms)" % (h, t.local_port, (time.perf_counter() - t0) * 1000),
              flush=True)

    try:
        addr = {h: "127.0.0.1:%d" % tunnels[h].local_port for h in tunnels}

        # ---- correctness: local CPU reference vs remote fleet ----
        ref_cfg = make_cfg(tmp / "ref.cfg", None, 1, extra={"numEigenThreadsPerModel": 1})
        ref = gtp(KATAGO_LOCAL, ref_cfg, RAW_SCRIPT)
        ref_txt = raw_text(ref.stdout)
        step("local CPU reference raw-NN", ref.returncode == 0 and "symmetry 7" in ref_txt,
             "%d bytes" % len(ref_txt))

        cfg2 = make_cfg(tmp / "two.cfg", ",".join(addr.values()), 2)
        r = gtp(KATAGO_CLIENT, cfg2, RAW_SCRIPT)
        step("raw-NN byte-identical over the 2-node fleet",
             r.returncode == 0 and raw_text(r.stdout) == ref_txt,
             "rc=%s identical=%s" % (r.returncode, raw_text(r.stdout) == ref_txt))

        # ---- per-node throughput (1 tunnel each) ----
        per_node = {}
        for h in ("weste", "westc"):
            cfg = make_cfg(tmp / ("one_%s.cfg" % h), addr[h], 1)
            b = bench(cfg, 8, 30)
            per_node[h] = b
            print("  %s alone, batch 8, 1 conn: %.1f evals/s (%.1f ms/batch)"
                  % (h, b["evals_s"], b["ms_per_batch"]), flush=True)
        results["per_node"] = per_node

        # ---- combined fleet ----
        combined = {}
        for threads in (2, 4):
            cfg = make_cfg(tmp / ("fleet_%d.cfg" % threads), ",".join(addr.values()), threads)
            sampler = GpuSampler(clients)
            sampler.start()
            b = bench(cfg, 8, 20)
            sampler.stop()
            combined[threads] = {"bench": b, "gpu": sampler.summary()}
            print("  fleet, %d conns: %.1f evals/s (%.1f ms/batch); GPU util %s"
                  % (threads, b["evals_s"], b["ms_per_batch"], sampler.summary()), flush=True)
        results["combined"] = combined

        # ---- batch sweep on one node ----
        sweep = {}
        for batch, iters in ((1, 20), (8, 30), (32, 12)):
            cfg = make_cfg(tmp / ("sweep_%d.cfg" % batch), addr["weste"], 1)
            b = bench(cfg, batch, iters)
            sweep[batch] = b
            print("  weste batch %-2d: %.1f evals/s (%.1f ms/batch)"
                  % (batch, b["evals_s"], b["ms_per_batch"]), flush=True)
        results["batch_sweep"] = sweep

        step("fleet raw-NN + throughput measured", True,
             "2 nodes, up to %.0f evals/s" % max(v["bench"]["evals_s"] for v in combined.values()))
    finally:
        for t in tunnels.values():
            try:
                t.stop()
            except Exception:
                pass
        for h, c in clients.items():
            try:
                sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
                c.close()
            except Exception:
                pass

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "cloud_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\nsaved -> %s" % (OUT / "cloud_results.json"))


if __name__ == "__main__":
    main()

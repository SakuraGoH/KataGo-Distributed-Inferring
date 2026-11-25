#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feasibility harness for KataGo's distributed inference backend.

The point of the harness is to emulate a *fleet of GPU machines* on a single
host: every "machine" is one `katago distserver` process listening on its own
port, exactly as it would on its own host.  From the client's point of view the
only difference between "3 processes on localhost" and "3 machines on the LAN"
is the value of `distServers`, which is what we vary.

Groups
------
correctness : raw-NN byte equality local-vs-remote, deterministic game equality,
              two engines sharing one fleet, 8-symmetry coverage.
robustness  : non-loopback bind/connect, handshake rejection, server down at
              startup, server killed mid-search, idle connection + concurrency.
gpu         : `-gpus 0,1,2` round-robin assignment across connections, and
              multiple connections sharing one GPU.
perf        : per-batch transport overhead vs batch size, 1->2->3 server
              scaling, RTT sensitivity through a delaying TCP proxy, and bytes
              per eval through a counting proxy.

Usage:  python dist_test/feasibility/harness.py [--groups all|correctness,...]
"""

import argparse
import atexit
import json
import os
import queue
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(r"C:\Users\10632\Desktop\KataGo-master")
SERVER_EXE = ROOT / "build-eigen" / "Release" / "katago.exe"   # normal backend = "the GPU machine"
CLIENT_EXE = ROOT / "build-dist" / "Release" / "katago.exe"    # USE_BACKEND=DISTRIBUTED
MODEL = Path(r"C:\Users\10632\Desktop\model.bin.gz")
BASE_CFG = ROOT / "cpp" / "configs" / "gtp_example.cfg"
OUT = ROOT / "dist_test" / "feasibility" / "out"
CFG_DIR = OUT / "cfg"
LOG_DIR = OUT / "logs"
PORT_BASE = 17800

SERVERS = []          # every Server we start, for cleanup
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def free_ports(n, start=PORT_BASE):
    ports = []
    p = start
    while len(ports) < n:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            ports.append(p)
        except OSError:
            pass
        finally:
            s.close()
        p += 1
    return ports


def make_cfg(name, visits=200, threads=1, servers=None, nn_threads=None,
             batch=None, extra=None, base=None):
    """Copy gtp_example.cfg, replace the active keys, append the dist keys."""
    text = (base or BASE_CFG).read_text(encoding="utf-8", errors="replace")
    repl = {
        "allowResignation": "false",
        "maxVisits": str(visits),
        "numSearchThreads": str(threads),
        "logAllGTPCommunication": "false",
        "logSearchInfo": "false",
        "nnRandomize": "false",
        # Both are needed for reproducible games: nnRandomize pins the input
        # symmetry, searchRandSeed pins the search RNG (node hashing etc.).
        "searchRandSeed": "12345",
    }
    if batch is not None:
        repl["nnMaxBatchSize"] = str(batch)
    out = []
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_]+)\s*=", line)
        if m and m.group(1) in repl:
            k = m.group(1)
            out.append("%s = %s" % (k, repl.pop(k)))
        else:
            out.append(line)
    for k, v in repl.items():
        out.append("%s = %s" % (k, v))
    if servers:
        out.append("distServers = %s" % servers)
    if nn_threads:
        out.append("numNNServerThreadsPerModel = %d" % nn_threads)
    for k, v in (extra or {}).items():
        out.append("%s = %s" % (k, v))
    path = CFG_DIR / (name + ".cfg")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


class Server:
    """One `katago distserver` process == one "GPU machine"."""

    def __init__(self, name, port, gpus=None, board=19, listen="127.0.0.1",
                 timeout=0, max_batch=256):
        self.name, self.port, self.listen_host = name, port, listen
        self.lines = []
        self._lock = threading.Lock()
        self.log_path = LOG_DIR / ("%s.log" % name)
        args = [str(SERVER_EXE), "distserver", "-model", str(MODEL),
                "-listen", "%s:%d" % (listen, port),
                "-boardsize", str(board), "-max-batch", str(max_batch)]
        if gpus is not None:
            args += ["-gpus", str(gpus)]
        if timeout:
            args += ["-timeout", str(timeout)]
        self.args = args
        self.proc = subprocess.Popen(
            args, cwd=str(OUT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0)
        SERVERS.append(self)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        if not self.wait_for("listening on", 30):
            raise RuntimeError("server %s failed to start: rc=%s log=%s"
                               % (name, self.proc.poll(), self.tail(10)))

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            with self._lock:
                self.lines.append(line)

    def lines_snapshot(self):
        with self._lock:
            return list(self.lines)

    def wait_for(self, substr, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(substr in ln for ln in self.lines_snapshot()):
                return True
            if self.proc.poll() is not None:
                time.sleep(0.1)
                return any(substr in ln for ln in self.lines_snapshot())
            time.sleep(0.05)
        return False

    def tail(self, n=8):
        return "\n".join(self.lines_snapshot()[-n:])

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                pass

    def connections(self):
        """(connId, gpu) pairs logged at handshake time."""
        out = []
        for ln in self.lines_snapshot():
            m = re.search(r"connection (\d+) from thread \(gpu (\S+),", ln)
            if m:
                out.append((int(m.group(1)), m.group(2)))
        return out

    def eval_batches(self):
        n = 0
        for ln in self.lines_snapshot():
            if "eval batch" in ln:
                n += 1
        return n


class Fleet:
    def __init__(self, n, prefix="srv", gpus=None, board=19, listen="127.0.0.1",
                 base_port=None, timeout=0, max_batch=256):
        ports = free_ports(n, base_port or PORT_BASE)
        self.servers = [Server("%s%d" % (prefix, i), ports[i], gpus=gpus,
                               board=board, listen=listen, timeout=timeout,
                               max_batch=max_batch) for i in range(n)]
        for s in self.servers:
            if not s.wait_for("listening on", 30):
                raise RuntimeError("fleet server %s not listening" % s.name)

    @property
    def addrs(self):
        return ",".join("%s:%d" % (s.listen_host, s.port) for s in self.servers)

    def stop(self):
        for s in self.servers:
            s.stop()

    def total(self, fn):
        return sum(getattr(s, fn)() if callable(getattr(s, fn)) else getattr(s, fn)
                   for s in self.servers)


def run_client(exe, cfg, script, timeout=600, tag="client"):
    args = [str(exe), "gtp", "-model", str(MODEL), "-config", str(cfg)]
    try:
        r = subprocess.run(args, input=script.encode(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, cwd=str(OUT), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {"rc": None, "stdout": (e.stdout or b"").decode("utf-8", "replace"),
                "stderr": (e.stderr or b"").decode("utf-8", "replace"),
                "timeout": True, "args": args}
    return {"rc": r.returncode, "stdout": r.stdout.decode("utf-8", "replace"),
            "stderr": r.stderr.decode("utf-8", "replace"), "timeout": False,
            "args": args}


def gtp(exe, cfg, script, timeout=600):
    return run_client(exe, cfg, script, timeout=timeout)


def check(name, ok, detail, **data):
    RESULTS["groups"].setdefault(CURRENT_GROUP, []).append(
        {"check": name, "ok": bool(ok), "detail": detail, **data})
    print(("[PASS] " if ok else "[FAIL] ") + name + "  " + detail, flush=True)
    return ok


CURRENT_GROUP = "misc"


def section(title):
    print("\n=== %s ===" % title, flush=True)


# ---------------------------------------------------------------------------
# GTP scripts
# ---------------------------------------------------------------------------

RAW_SCRIPT = """boardsize 19
komi 6.5
kata-raw-nn b all
play b Q16
play w D16
kata-raw-nn w all
play b R4
play w D4
kata-raw-nn b 3
kata-raw-nn w 5
quit
"""

GAME_SCRIPT = """boardsize 19
komi 6.5
genmove b
genmove w
genmove b
genmove w
genmove b
genmove w
quit
"""


def err_lines(stderr, limit=2):
    """Last informative lines of a KataGo stderr dump (errors are multi-line)."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    keep = [ln for ln in lines if "ERROR" in ln or "Error" in ln or "error" in ln] or lines
    return keep[-limit:]


def move_lines(stdout):
    return "\n".join(ln for ln in stdout.splitlines() if ln[:1] in ("=", "!"))


# ---------------------------------------------------------------------------
# group: correctness
# ---------------------------------------------------------------------------

def group_correctness():
    section("correctness: raw NN and deterministic games, local vs remote")
    cfg_local = make_cfg("ana_local", batch=8, extra={"numEigenThreadsPerModel": 1})
    ref = gtp(SERVER_EXE, cfg_local, RAW_SCRIPT, timeout=300)
    if ref["rc"] != 0:
        check("local reference raw-nn", False, ref["stderr"][-400:])
        return
    ref_raw = ref["stdout"]
    check("local reference raw-nn", "symmetry 0" in ref_raw,
          "%d bytes of raw-NN output" % len(ref_raw))

    # one server, one NN thread -- the minimal multi-machine setup
    f1 = Fleet(1, prefix="c1", base_port=PORT_BASE)
    try:
        cfg1 = make_cfg("ana_d1", batch=8, servers=f1.addrs, nn_threads=1)
        d1 = gtp(CLIENT_EXE, cfg1, RAW_SCRIPT, timeout=300)
        check("raw-NN byte-identical via 1 remote server",
              d1["rc"] == 0 and d1["stdout"] == ref_raw,
              "rc=%s, identical=%s" % (d1["rc"], d1["stdout"] == ref_raw),
              server_batches=f1.total("eval_batches"))

        # A client that asks for NCHW input (inputsUseNHWC=false) forces the server to
        # transpose: the direction has to follow the CLIENT's layout, not be assumed.
        cfg_nchw = make_cfg("ana_nchw", batch=8, servers=f1.addrs, nn_threads=1,
                            extra={"inputsUseNHWC": "false"})
        dn = gtp(CLIENT_EXE, cfg_nchw, RAW_SCRIPT, timeout=300)
        check("NCHW client -> NHWC-only server: transpose direction correct",
              dn["rc"] == 0 and dn["stdout"] == ref_raw,
              "rc=%s, identical=%s" % (dn["rc"], dn["stdout"] == ref_raw))

        # three servers, three NN threads: every thread must get identical resuls
        f3 = Fleet(3, prefix="c3", base_port=PORT_BASE + 20)
        try:
            cfg3 = make_cfg("ana_d3", batch=8, servers=f3.addrs, nn_threads=3)
            d3 = gtp(CLIENT_EXE, cfg3, RAW_SCRIPT, timeout=300)
            check("raw-NN byte-identical via 3-server fleet (3 NN threads)",
                  d3["rc"] == 0 and d3["stdout"] == ref_raw,
                  "rc=%s, identical=%s, per-server batches=%s"
                  % (d3["rc"], d3["stdout"] == ref_raw,
                     [s.eval_batches() for s in f3.servers]))
            per_server = [s.eval_batches() for s in f3.servers]
            # Batches are pulled from one shared queue, so a very short run can leave one
            # connection with nothing to do; what matters is that the fleet is used.
            check("work is spread over multiple machines",
                  sum(per_server) > 0 and sum(1 for n in per_server if n > 0) >= 2,
                  "batches per server = %s (a short run may leave one connection idle)"
                  % per_server)
        finally:
            f3.stop()
    finally:
        f1.stop()

    # deterministic game: same config, remote vs local must pick the same moves
    cfg_gl = make_cfg("game_local", visits=60, threads=1, batch=8,
                      extra={"numEigenThreadsPerModel": 1})
    gl = gtp(SERVER_EXE, cfg_gl, GAME_SCRIPT, timeout=600)
    check("local reference game", gl["rc"] == 0 and gl["stdout"].count("=") >= 6,
          "moves=%r" % move_lines(gl["stdout"]).replace("\n", " "))

    f1 = Fleet(1, prefix="g1", base_port=PORT_BASE + 40)
    try:
        cfg_gl_d = make_cfg("game_dist1", visits=60, threads=1, batch=8,
                            servers=f1.addrs, nn_threads=1)
        gd = gtp(CLIENT_EXE, cfg_gl_d, GAME_SCRIPT, timeout=600)
        check("6-move game identical over remote inference (fixed seeds)",
              gd["rc"] == 0 and move_lines(gd["stdout"]) == move_lines(gl["stdout"]),
              "remote=%r" % move_lines(gd["stdout"]).replace("\n", " "))
    finally:
        f1.stop()

    # two independent engines sharing the same two-machine fleet
    section("correctness: two engines share one two-machine fleet")
    f2 = Fleet(2, prefix="m2", base_port=PORT_BASE + 60)
    try:
        cfg_a = make_cfg("multi_a", batch=8, servers=f2.addrs, nn_threads=2)
        cfg_b = make_cfg("multi_b", batch=8, servers=f2.addrs, nn_threads=2)
        res = {}

        def worker(tag, cfg):
            res[tag] = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=300)

        ta = threading.Thread(target=worker, args=("a", cfg_a))
        tb = threading.Thread(target=worker, args=("b", cfg_b))
        ta.start(); tb.start(); ta.join(); tb.join()
        ok = all(r["rc"] == 0 and r["stdout"] == ref_raw for r in res.values())
        check("both concurrent engines get byte-identical results", ok,
              "rc_a=%s rc_b=%s, per-server batches=%s"
              % (res["a"]["rc"], res["b"]["rc"],
                 [s.eval_batches() for s in f2.servers]))
        check("both machines served both engines",
              all(s.eval_batches() > 0 for s in f2.servers),
              "batches per server = %s" % [s.eval_batches() for s in f2.servers])
    finally:
        f2.stop()


# ---------------------------------------------------------------------------
# group: robustness
# ---------------------------------------------------------------------------

def group_robustness():
    section("robustness: LAN address, handshake validation, failures")

    # -- non-loopback interface ------------------------------------------------
    lan_ip = None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object "
                              "{$_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*'}"
                              " | Select-Object -First 1).IPAddress"],
                             capture_output=True, text=True, timeout=20)
        lan_ip = out.stdout.strip() or None
    except Exception:
        lan_ip = None

    if lan_ip:
        try:
            f = Fleet(1, prefix="lan", base_port=PORT_BASE + 80, listen=lan_ip)
            cfg = make_cfg("ana_lan", batch=8, servers=f.addrs, nn_threads=1)
            r = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=300)
            check("server bound to LAN address %s, client over it" % lan_ip,
                  r["rc"] == 0 and "symmetry 7" in r["stdout"],
                  "rc=%s" % r["rc"], server_batches=f.total("eval_batches"))
            f.stop()
        except Exception as e:
            check("server bound to LAN address %s" % lan_ip, False, repr(e))
    else:
        check("LAN address test", False, "no non-loopback IPv4 found")

    # -- handshake validation --------------------------------------------------
    f = Fleet(1, prefix="hs", base_port=PORT_BASE + 100)
    try:
        s = f.servers[0]
        try:
            sock = socket.create_connection(("127.0.0.1", s.port), timeout=10)
            sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            sock.settimeout(10)
            try:
                data = sock.recv(4096)
                outcome = "error status %s: %r" % (
                    int.from_bytes(data[:4], "little"), data[8:].decode(errors="replace"))
                ok = len(data) >= 4 and int.from_bytes(data[:4], "little") == 1
            except ConnectionResetError as e:
                # Unread request bytes + immediate close makes Windows send RST and
                # drop the error message; the handshake is still refused.
                outcome = "connection reset (error message discarded): %s" % e
                ok = True
            sock.close()
            check("bad magic is refused by the handshake", ok, outcome)
        except Exception as e:
            check("bad magic is refused by the handshake", False, repr(e))

        cfg_ok = make_cfg("hs_ok", batch=8, servers=f.addrs, nn_threads=1)
        r = gtp(CLIENT_EXE, cfg_ok, RAW_SCRIPT, timeout=300)
        check("server keeps serving after a bogus client",
              r["rc"] == 0 and "symmetry 0" in r["stdout"], "rc=%s" % r["rc"])

        # board-size mismatch must fail the handshake, not corrupt tensors
        f9 = Fleet(1, prefix="bs9", base_port=PORT_BASE + 120, board=9)
        try:
            cfg_bad = make_cfg("hs_board", batch=8, servers=f9.addrs, nn_threads=1)
            rb = gtp(CLIENT_EXE, cfg_bad, RAW_SCRIPT, timeout=120)
            check("board-size mismatch rejected at handshake",
                  rb["rc"] not in (0, None) and "board size mismatch" in rb["stderr"],
                  "rc=%s stderr=%r" % (rb["rc"], err_lines(rb["stderr"])))
        finally:
            f9.stop()
    finally:
        f.stop()

    # -- server absent at startup ---------------------------------------------
    dead_port = free_ports(1, PORT_BASE + 140)[0]
    cfg_dead = make_cfg("dead", batch=8, servers="127.0.0.1:%d" % dead_port, nn_threads=1)
    t0 = time.time()
    rd = gtp(CLIENT_EXE, cfg_dead, RAW_SCRIPT, timeout=120)
    check("client fails fast with a clear error if a server is down",
          rd["rc"] not in (0, None) and "could not connect" in rd["stderr"],
          "rc=%s after %.2fs: %r" % (rd["rc"], time.time() - t0, err_lines(rd["stderr"])))

    # -- client disappearing must not kill the server --------------------------
    f = Fleet(1, prefix="idle", base_port=PORT_BASE + 160)
    try:
        cfg = make_cfg("idle", batch=8, servers=f.addrs, nn_threads=1)
        r = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=120)
        time.sleep(0.3)
        r2 = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=120)
        check("server survives client disconnect and serves the next client",
              r["rc"] == 0 and r2["rc"] == 0 and f.servers[0].proc.poll() is None,
              "two sequential clients, server alive=%s" % (f.servers[0].proc.poll() is None))
    finally:
        f.stop()

    # -- misconfiguration: batch cap mismatch ---------------------------------
    f = Fleet(1, prefix="mb", base_port=PORT_BASE + 165, max_batch=4)
    try:
        cfg = make_cfg("maxbatch", batch=8, servers=f.addrs, nn_threads=1)
        try:
            bench_nn(CLIENT_EXE, cfg, 8, iters=1, warmup=0, timeout=120)
            check("client nnMaxBatchSize > server -max-batch fails fast", False,
                  "benchmark ran even though the server caps batches at 4")
        except RuntimeError as e:
            msg = str(e)
            check("client nnMaxBatchSize > server -max-batch fails fast",
                  "exceeds this server's -max-batch" in msg,
                  msg[-220:] if len(msg) > 220 else msg)
    finally:
        f.stop()

    # -- the client binary must not pretend to be a server --------------------
    r = subprocess.run([str(CLIENT_EXE), "distserver", "-model", str(MODEL)],
                       cwd=str(OUT), capture_output=True, timeout=60)
    check("client-only (-DUSE_BACKEND=DISTRIBUTED) build refuses to run distserver",
          r.returncode != 0 and b"cannot serve evals" in r.stderr,
          "rc=%s" % r.returncode)

    # -- server death during search -------------------------------------------
    f = Fleet(1, prefix="kill", base_port=PORT_BASE + 180)
    try:
        cfg = make_cfg("kill", visits=5000, threads=1, batch=8,
                       servers=f.addrs, nn_threads=1)
        p = subprocess.Popen([str(CLIENT_EXE), "gtp", "-model", str(MODEL),
                              "-config", str(cfg)], cwd=str(OUT),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.stdin.write(b"boardsize 19\ngenmove b\n")
        p.stdin.flush()
        time.sleep(2.0)
        f.servers[0].stop()
        try:
            out, err = p.communicate(timeout=60)
            rc, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            rc, timed_out = None, True
        err_txt = err.decode("utf-8", "replace").strip()
        check("server killed mid-search: client errors out instead of hanging",
              (not timed_out) and rc != 0,
              "rc=%s timed_out=%s stderr=%r"
              % (rc, timed_out, err_lines(err_txt)))
    finally:
        f.stop()

    # -- concurrency: a second engine is served while the first is still busy --
    f = Fleet(1, prefix="conc", base_port=PORT_BASE + 200)
    try:
        cfg_long = make_cfg("conc_long", visits=4000, threads=1, batch=8,
                            servers=f.addrs, nn_threads=1)
        cfg_short = make_cfg("conc_short", batch=8, servers=f.addrs, nn_threads=1)
        p = subprocess.Popen([str(CLIENT_EXE), "gtp", "-model", str(MODEL),
                              "-config", str(cfg_long)], cwd=str(OUT),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.stdin.write(b"boardsize 19\ngenmove b\n")
        p.stdin.flush()
        time.sleep(1.5)
        t0 = time.time()
        rs = gtp(CLIENT_EXE, cfg_short, RAW_SCRIPT, timeout=120)
        served_while_busy = (p.poll() is None)
        p.kill()
        p.communicate()
        check("second engine served while the first is mid-search",
              rs["rc"] == 0 and served_while_busy,
              "short client rc=%s in %.2fs, long client still running=%s"
              % (rs["rc"], time.time() - t0, served_while_busy))
    finally:
        f.stop()


# ---------------------------------------------------------------------------
# group: gpu
# ---------------------------------------------------------------------------

def group_gpu():
    section("multi-GPU: per-connection device assignment")
    f = Fleet(1, prefix="gpu", base_port=PORT_BASE + 220, gpus="0,1,2")
    try:
        cfg = make_cfg("gpu3", batch=4, servers=f.addrs, nn_threads=6)
        r = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=300)
        # Connections are assigned gpu = connId % gpus at accept time; the log lines
        # themselves can appear out of order because each thread logs after its handshake.
        conns = sorted(f.servers[0].connections())
        gpus = [g for _, g in conns]
        expect = ["0", "1", "2", "0", "1", "2"]
        check("6 connections round-robin over -gpus 0,1,2",
              r["rc"] == 0 and gpus[:6] == expect,
              "connId->gpu=%s expected_prefix=%s rc=%s" % (conns[:6], expect, r["rc"]))
        check("all three devices actually served work",
              r["rc"] == 0 and len(set(gpus)) == 3,
              "distinct devices=%s" % sorted(set(gpus)))
    finally:
        f.stop()

    f = Fleet(1, prefix="gpu1", base_port=PORT_BASE + 240, gpus="0")
    try:
        cfg = make_cfg("gpu1", batch=4, servers=f.addrs, nn_threads=4)
        r = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=300)
        gpus = [g for _, g in f.servers[0].connections()]
        check("all connections pinned to one GPU when -gpus 0",
              r["rc"] == 0 and set(gpus) == {"0"}, "assigned=%s" % gpus)
    finally:
        f.stop()

    # two processes, one GPU each: the "one distserver per GPU" deployment
    f = Fleet(2, prefix="gpu2x", base_port=PORT_BASE + 260)
    try:
        f.servers[0].stop(); f.servers[1].stop()
        s0 = Server("gpu2x0", free_ports(1, PORT_BASE + 260)[0], gpus="0")
        s1 = Server("gpu2x1", free_ports(1, PORT_BASE + 261)[0], gpus="1")
        addrs = "127.0.0.1:%d,127.0.0.1:%d" % (s0.port, s1.port)
        cfg = make_cfg("gpu2x", batch=4, servers=addrs, nn_threads=2)
        r = gtp(CLIENT_EXE, cfg, RAW_SCRIPT, timeout=300)
        check("one process per GPU (gpus=0 / gpus=1) both serve",
              r["rc"] == 0 and s0.eval_batches() > 0 and s1.eval_batches() > 0,
              "batches s0=%d s1=%d" % (s0.eval_batches(), s1.eval_batches()))
    finally:
        f.stop()


# ---------------------------------------------------------------------------
# group: perf
# ---------------------------------------------------------------------------

def bench_nn(exe, cfg, batch, iters=4, warmup=1, timeout=900):
    args = [str(exe), "benchmarknn", "-model", str(MODEL), "-config", str(cfg),
            "-iterations", str(iters), "-warmup", str(warmup),
            "-batch-size", str(batch), "-json"]
    r = subprocess.run(args, cwd=str(OUT), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    line = [ln for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.strip().startswith("{")]
    if r.returncode != 0 or not line:
        raise RuntimeError("benchmarknn failed rc=%s stderr=%s" % (
            r.returncode, r.stderr.decode("utf-8", "replace")[-800:]))
    return json.loads(line[-1])


def eval_times_since(server, line_idx, count):
    """The last `count` server-side 'eval batch ... in Xs' durations logged since line_idx."""
    times = []
    for ln in server.lines_snapshot()[line_idx:]:
        m = re.search(r"eval batch (\d+) in ([\d.]+)s", ln)
        if m:
            times.append(float(m.group(2)))
    return times[-count:] if count > 0 else times


def group_perf():
    section("perf: client round trip vs server-side compute, per batch size")
    # Comparing against server-side compute measured around the SAME eval isolates
    # transport from CPU-frequency/core-placement noise between two processes.
    f = Fleet(1, prefix="perf1", base_port=PORT_BASE + 280)
    srv = f.servers[0]
    try:
        overhead = []
        for batch in (1, 8, 32):
            iters = 4
            cfg_dist = make_cfg("p_dist_b%d" % batch, batch=batch,
                                servers=f.addrs, nn_threads=1)
            n0 = len(srv.lines_snapshot())
            r = bench_nn(CLIENT_EXE, cfg_dist, batch, iters=iters, warmup=1)
            cli_ms = r["perThreadMedianMs"][0]
            st = eval_times_since(srv, n0, iters)
            srv_ms = statistics.median(st) * 1000.0 if st else float("nan")
            row = {"batch": batch, "client_ms": cli_ms, "server_compute_ms": srv_ms,
                   "transport_ms": cli_ms - srv_ms,
                   "transport_frac": (cli_ms - srv_ms) / cli_ms,
                   "server_samples": len(st)}
            overhead.append(row)
            print("  batch %-3d server compute %8.1f ms  client round trip %8.1f ms  "
                  "transport %6.2f ms (%.1f%%)"
                  % (batch, srv_ms, cli_ms, cli_ms - srv_ms,
                     100.0 * (cli_ms - srv_ms) / cli_ms), flush=True)
        RESULTS["groups"]["perf_overhead"] = overhead
        check("per-eval transport cost measured against server-side compute", True,
              "; ".join("B%d:%.1fms" % (o["batch"], o["transport_ms"]) for o in overhead))
    finally:
        f.stop()

    section("perf: adding machines vs adding connections (batch 8, best of 2 runs)")
    # Same total number of client connections on one server ("single") versus spread
    # over n servers ("split"): equal core count on both sides, so a gap would be a
    # transport/dispatch cost rather than a CPU ceiling.
    fleet1 = Fleet(1, prefix="s1", base_port=PORT_BASE + 300)
    fleet3 = Fleet(3, prefix="s3", base_port=PORT_BASE + 320)
    try:
        samples = {}

        def once(kind, n, rep):
            if kind == "split":
                addrs = ",".join("%s:%d" % (s.listen_host, s.port)
                                 for s in fleet3.servers[:n])
                cfg = make_cfg("scale_split_%d_%d" % (n, rep), batch=8,
                               servers=addrs, nn_threads=n)
                exe = CLIENT_EXE
            elif kind == "single":
                cfg = make_cfg("scale_single_%d_%d" % (n, rep), batch=8,
                               servers=fleet1.addrs, nn_threads=n)
                exe = CLIENT_EXE
            else:
                cfg = make_cfg("scale_local_%d_%d" % (n, rep), batch=8,
                               extra={"numEigenThreadsPerModel": n})
                exe = SERVER_EXE
            r = bench_nn(exe, cfg, 8, iters=4, warmup=1)
            return r["actualWallNNEvalsPerSec"]

        for rep in range(2):
            for n in (1, 2, 3):
                for kind in ("split", "single", "local"):
                    samples.setdefault((kind, n), []).append(once(kind, n, rep))

        rows = []
        base = max(samples[("split", 1)])
        for n in (1, 2, 3):
            split = max(samples[("split", n)])
            single = max(samples[("single", n)])
            local = max(samples[("local", n)])
            row = {"machines": n, "conns": n,
                   "split_evals_per_s": split, "split_samples": samples[("split", n)],
                   "single_evals_per_s": single, "single_samples": samples[("single", n)],
                   "local_evals_per_s": local, "local_samples": samples[("local", n)],
                   "split_vs_single": split / single,
                   "speedup_vs_1machine": split / base}
            rows.append(row)
            print("  %d conn(s): split over %d machine(s) %6.2f evals/s | all on 1 machine "
                  "%6.2f | local %6.2f | split/single %.2f | growth x%.2f"
                  % (n, n, split, single, local, split / single, split / base), flush=True)
        RESULTS["groups"]["perf_scaling"] = rows
        check("splitting the same connections over more machines costs nothing",
              all(r["split_vs_single"] >= 0.85 for r in rows),
              "; ".join("%d conn(s): %.0f%% of one-machine throughput"
                        % (r["conns"], 100.0 * r["split_vs_single"]) for r in rows))
        check("aggregate throughput grows with the number of machines",
              rows[-1]["split_evals_per_s"] > 1.25 * base,
              "1->%d machines: %.2f -> %.2f evals/s (x%.2f); a CPU stand-in with 4 P-cores "
              "and 8 E-cores cannot be expected to show linear growth"
              % (rows[-1]["machines"], base, rows[-1]["split_evals_per_s"],
                 rows[-1]["speedup_vs_1machine"]))
    finally:
        fleet1.stop()
        fleet3.stop()

    section("perf: network latency sensitivity (delaying proxy, RTT = 2 x one-way)")
    f = Fleet(1, prefix="wan", base_port=PORT_BASE + 400)
    srv = f.servers[0]
    try:
        for batch, iters in ((1, 8), (8, 6)):
            for rtt_ms in (0, 2, 10, 40):
                proxy = Proxy("127.0.0.1", srv.port, delay_s=rtt_ms / 2000.0)
                try:
                    cfg = make_cfg("wan_b%d_r%d" % (batch, rtt_ms), batch=batch,
                                   servers="127.0.0.1:%d" % proxy.port, nn_threads=1)
                    n0 = len(srv.lines_snapshot())
                    r = bench_nn(CLIENT_EXE, cfg, batch, iters=iters, warmup=1)
                    cli_ms = r["perThreadMedianMs"][0]
                    st = eval_times_since(srv, n0, iters)
                    srv_ms = statistics.median(st) * 1000.0 if st else float("nan")
                    row = {"batch": batch, "rtt_ms": rtt_ms,
                           "client_ms": cli_ms, "server_compute_ms": srv_ms,
                           "transport_ms": cli_ms - srv_ms,
                           "evals_per_s": r["actualWallNNEvalsPerSec"]}
                    RESULTS["groups"].setdefault("perf_wan", []).append(row)
                    print("  batch %-2d RTT %2d ms: %7.2f evals/s  "
                          "(transport %6.2f ms, compute %7.1f ms)"
                          % (batch, rtt_ms, r["actualWallNNEvalsPerSec"],
                             cli_ms - srv_ms, srv_ms), flush=True)
                finally:
                    proxy.stop()
    finally:
        f.stop()

    section("perf: bytes on the wire per eval (counting proxy)")
    f = Fleet(1, prefix="bytes", base_port=PORT_BASE + 420)
    try:
        for batch in (1, 8):
            proxy = Proxy("127.0.0.1", f.servers[0].port, delay_s=0.0, count_bytes=True)
            try:
                cfg = make_cfg("bytes_b%d" % batch, batch=batch,
                               servers="127.0.0.1:%d" % proxy.port, nn_threads=1)
                iters, warmup = 4, 1
                r = bench_nn(CLIENT_EXE, cfg, batch, iters=iters, warmup=warmup)
                # The proxy also sees the untimed warmup pass.
                evals = (iters + warmup) * batch
                up = proxy.bytes_c2s / evals
                down = proxy.bytes_s2c / evals
                row = {"batch": batch, "up_bytes_per_eval": up,
                       "down_bytes_per_eval": down}
                RESULTS["groups"].setdefault("perf_bytes", []).append(row)
                print("  batch %-2d: %.0f B/eval up, %.0f B/eval down, total %.1f KB/eval"
                      % (batch, up, down, (up + down) / 1024.0), flush=True)
            finally:
                proxy.stop()
    finally:
        f.stop()


# ---------------------------------------------------------------------------
# delaying / counting TCP proxy
# ---------------------------------------------------------------------------

class Proxy:
    """TCP proxy that delays every chunk by `delay_s` one way (=> RTT ~= 2*delay)."""

    def __init__(self, target_host, target_port, delay_s=0.0, count_bytes=True):
        self.delay = delay_s
        self.target = (target_host, target_port)
        self.bytes_c2s = 0
        self.bytes_s2c = 0
        self._n = 0
        self._lock = threading.Lock()
        self._stop = False
        self._socks = []
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(64)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                c, _ = self._srv.accept()
            except OSError:
                return
            try:
                s = socket.create_connection(self.target)
            except OSError:
                c.close()
                continue
            for x in (c, s):
                x.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socks += [c, s]
            threading.Thread(target=self._pipe, args=(c, s, "c2s"), daemon=True).start()
            threading.Thread(target=self._pipe, args=(s, c, "s2c"), daemon=True).start()

    def _pipe(self, src, dst, direction):
        q = queue.Queue()

        def reader():
            try:
                while True:
                    d = src.recv(262144)
                    if not d:
                        break
                    q.put((time.monotonic() + self.delay, d))
            except OSError:
                pass
            q.put((time.monotonic() + self.delay, None))

        def writer():
            while True:
                due, d = q.get()
                if d is None:
                    break
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                try:
                    dst.sendall(d)
                except OSError:
                    break
                with self._lock:
                    if direction == "c2s":
                        self.bytes_c2s += len(d)
                    else:
                        self.bytes_s2c += len(d)
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=writer, daemon=True).start()

    def stop(self):
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass
        for s in self._socks:
            try:
                s.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def cleanup():
    for s in SERVERS:
        s.stop()


def main():
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="all")
    args = ap.parse_args()
    groups = (["correctness", "robustness", "gpu", "perf"] if args.groups == "all"
              else [g.strip() for g in args.groups.split(",") if g.strip()])

    global CURRENT_GROUP
    OUT.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    atexit.register(cleanup)

    # refuse to run with a stale katago fleet holding ports
    for g in groups:
        CURRENT_GROUP = g
        try:
            {"correctness": group_correctness, "robustness": group_robustness,
             "gpu": group_gpu, "perf": group_perf}[g]()
        except Exception:
            print("[ERROR] group %s crashed:\n%s" % (g, traceback.format_exc()), flush=True)
            RESULTS["groups"].setdefault(g, []).append(
                {"check": "group " + g, "ok": False, "detail": "crashed"})
        finally:
            cleanup()
            SERVERS.clear()

    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str),
                                      encoding="utf-8")
    npass = sum(1 for g in RESULTS["groups"].values() for c in g
                if isinstance(c, dict) and c.get("ok"))
    nfail = sum(1 for g in RESULTS["groups"].values() for c in g
                if isinstance(c, dict) and c.get("ok") is False)
    print("\n==== %d passed, %d failed; raw results in %s"
          % (npass, nfail, OUT / "results.json"), flush=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

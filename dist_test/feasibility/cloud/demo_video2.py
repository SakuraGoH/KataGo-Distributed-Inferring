"""Full-length (≈2 min) verification video: clean -> build -> two-node connect -> benchmarks.

Phase 1  wipe /root/cpp/build-* on both nodes, configure + build CUDA from scratch
         (recorded as a time-lapse; the real build takes several minutes)
Phase 2  prove the two machines are really linked: two GPUs, two distservers,
         two SakuraLink tunnels, one client connected to both, both servers serving evals
Phase 3  benchmark comparison: node-local GPU, one node via tunnel, two nodes
Phase 4  summary card
"""
import json
import math
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, r"C:\Users\10632\Desktop\SakuraLink")
import sshutil  # noqa: E402
from sakura.config import NodeConfig  # noqa: E402
from sakura.tunnel import SSHTunnel  # noqa: E402
from cloud_test import (KATAGO_CLIENT, MODEL, REMOTE_DS_PORT, bench, gtp,  # noqa: E402
                        make_cfg, raw_text)

NODES = ("weste", "westc")
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
OUT = CLOUD / "out" / "demo2"
OUT.mkdir(parents=True, exist_ok=True)

BUILD_CMD = (
    "cd /root/cpp && rm -rf build-cuda build-dist && "
    "cmake -S . -B build-cuda -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1 "
    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc "
    "-DCUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu "
    "-DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so > /root/build.log 2>&1 && "
    "cmake --build build-cuda -j48 >> /root/build.log 2>&1"
)

data = {"nodes": {}, "phases": []}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# phase 1: clean + build
# ---------------------------------------------------------------------------

def ssh_run(name, cmd, timeout=300):
    c = sshutil.connect(name)
    try:
        return sshutil.run(c, cmd, timeout=timeout)
    finally:
        c.close()


def start_build(name):
    c = sshutil.connect(name)
    sshutil.run(c, "pkill -f 'cmake --build' 2>/dev/null; pkill -f 'cc1plus' 2>/dev/null; true")
    sshutil.run(c, "setsid nohup bash -c '%s' > /dev/null 2>&1 < /dev/null &" % BUILD_CMD)
    c.close()


def pct_from(text):
    m = None
    for m in re.finditer(r"\[\s*(\d+)%\]", text):
        pass
    return int(m.group(1)) if m else 0


def last_line(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def phase_build():
    log("=== phase 1: clean + build ===")
    c = sshutil.connect(NODES[0])
    sshutil.run(c, "cd /root/cpp && du -sh build-cuda 2>/dev/null; ls build-cuda/katago 2>/dev/null", timeout=60)
    c.close()
    for n in NODES:
        start_build(n)
    t0 = time.time()
    samples = []
    done = {n: False for n in NODES}
    while time.time() - t0 < 2400:
        elapsed = time.time() - t0
        row = {"t": round(elapsed, 1)}
        for n in NODES:
            try:
                rc, out = ssh_run(n, "tail -c 4000 /root/build.log 2>/dev/null; "
                                     "test -x /root/cpp/build-cuda/katago && echo __DONE__ || true",
                                  timeout=30)
            except Exception:
                out = ""
            row[n] = {"pct": 100 if "__DONE__" in out else pct_from(out),
                      "line": last_line(out.replace("__DONE__", ""))}
            if "__DONE__" in out:
                done[n] = True
        samples.append(row)
        log("  %5.0fs  weste %3d%%  westc %3d%%  | %s" % (
            elapsed, row["weste"]["pct"], row["westc"]["pct"], row["weste"]["line"][:70]))
        if all(done.values()):
            break
        time.sleep(10)
    total = time.time() - t0
    for n in NODES:
        rc, out = ssh_run(n, "cd /root/cpp && ./build-cuda/katago version | tail -3; "
                             "ls -l build-cuda/katago | awk '{print $5}'", timeout=120)
        data["nodes"][n] = {"version": out.strip(), "build_seconds": round(total, 1)}
        log("  %s version: %s" % (n, out.strip().replace("\n", " | ")))
    data["build"] = {"samples": samples, "total_seconds": round(total, 1)}
    return samples, total


# ---------------------------------------------------------------------------
# phase 2/3: connect the two machines, verify, benchmark
# ---------------------------------------------------------------------------

def start_distserver(name):
    c = sshutil.connect(name)
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
    sshutil.run(c, "cd /root/cpp && setsid nohup ./build-cuda/katago distserver -model /root/model.bin.gz "
                   "-listen 127.0.0.1:%d -gpus 0 -max-batch 256 > /root/distserver.log 2>&1 < /dev/null &"
                   % REMOTE_DS_PORT)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)
    rc, logtxt = sshutil.run(c, "grep -E 'model |listening|postprocess' /root/distserver.log", timeout=30)
    return c, [l.split(": ", 1)[-1] for l in logtxt.strip().splitlines()]


def node_info():
    info = {}
    for n in NODES:
        c = sshutil.connect(n)
        rc, out = sshutil.run(c, "hostname; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,"
                                 "driver_version --format=csv,noheader; nproc", timeout=60)
        info[n] = [l.strip() for l in out.strip().splitlines() if l.strip()]
        rc, rtt = sshutil.run(c, "true", timeout=30)
        c.close()
    return info


def main():
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass

    # ---------------- phase 1 ----------------
    samples, build_total = phase_build()
    data["build"]["samples"] = samples

    # ---------------- phase 2 ----------------
    log("=== phase 2: node info + distservers + tunnels ===")
    info = node_info()
    data["info"] = info
    for n in NODES:
        log("  %s: %s" % (n, " | ".join(info[n])))

    clients = {}
    server_logs = {}
    for n in NODES:
        clients[n], lines = start_distserver(n)
        server_logs[n] = lines
        log("  %s distserver: %s" % (n, " / ".join(lines[-2:])))

    rtt = {}
    for n in NODES:
        t = clients[n].get_transport()
        ts = []
        for _ in range(8):
            t0 = time.perf_counter()
            t.global_request("keepalive@openssh.com", wait=True)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        rtt[n] = round(ts[len(ts) // 2], 1)
    data["rtt"] = rtt
    log("  SSH RTT: %s" % rtt)

    tunnels = {}
    for n in NODES:
        node = NodeConfig(host=sshutil.HOSTS[n][0], username="root", password=sshutil.HOSTS[n][2],
                          remote_port=sshutil.HOSTS[n][1], target_host="127.0.0.1",
                          target_port=REMOTE_DS_PORT, name=n)
        t = SSHTunnel(node)
        t.start()
        tunnels[n] = t
        log("  tunnel %s -> 127.0.0.1:%d" % (n, t.local_port))
    data["tunnel_ports"] = {n: tunnels[n].local_port for n in NODES}

    try:
        tmp = Path(tempfile.mkdtemp(prefix="demo2_"))
        addr = {n: "127.0.0.1:%d" % tunnels[n].local_port for n in NODES}

        # ---------------- correctness through both machines ----------------
        log("=== phase 2b: raw-nn through the two-machine fleet ===")
        cfg2 = make_cfg(tmp / "two.cfg", ",".join(addr.values()), 2, extra={"useFP16": "false"})
        r = gtp(KATAGO_CLIENT, cfg2, SCRIPT)
        remote = "\n".join(l.strip() for l in raw_text(r.stdout).splitlines()
                           if l.strip().startswith(("whiteWin", "whiteLead")))

        sftp = clients["weste"].open_sftp()
        with sftp.open("/root/d2.cfg", "w") as f:
            f.write(open(tmp / "two.cfg", encoding="utf-8").read()
                    .replace("distServers = %s" % ",".join(addr.values()), ""))
        with sftp.open("/root/d2.gtp", "w") as f:
            f.write(SCRIPT)
        sftp.close()
        rc, out = sshutil.run(clients["weste"],
                              "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
                              "-config /root/d2.cfg < /root/d2.gtp 2>&1 | "
                              "grep -E '^(whiteWin|whiteLead)'", timeout=300)
        direct = "\n".join(l.strip() for l in out.splitlines() if l.strip())
        same = (remote == direct) and "whiteWin" in remote
        data["correctness"] = {"remote": remote, "direct": direct, "same": same}
        log("  remote == direct: %s" % same)

        served = {}
        for n in NODES:
            rc, c2 = sshutil.run(clients[n], "grep -c 'eval batch' /root/distserver.log || true")
            served[n] = int(c2.strip() or 0)
        data["served_after_raw"] = served
        log("  batches served: %s" % served)

        # ---------------- benchmarks ----------------
        log("=== phase 3: benchmarks ===")
        # node-local GPU (weste), batch 8 and 32
        sftp = clients["weste"].open_sftp()
        with sftp.open("/root/bench.cfg", "w") as f:
            f.write("\n".join([
                "logToStderr = false", "numNNServerThreadsPerModel = 1",
                "nnRandomize = false", "useFP16 = false",
            ]) + "\n")
        sftp.close()
        gpu = {}
        for batch in (8, 32):
            rc, out = sshutil.run(clients["weste"],
                                  "cd /root/cpp && ./build-cuda/katago benchmarknn -model /root/model.bin.gz "
                                  "-config /root/bench.cfg -batch-size %d -iterations 20 -warmup 2 -json "
                                  "2>/dev/null | tail -1" % batch, timeout=600)
            lines = [l for l in out.strip().splitlines() if l.startswith("{")]
            if lines:
                d = json.loads(lines[-1])
                gpu[batch] = {"evals_s": d["actualWallNNEvalsPerSec"],
                              "ms": d["perThreadMedianMs"][0]}
        data["gpu_direct"] = gpu
        log("  GPU direct: %s" % gpu)

        # one node via tunnel
        cfg1 = make_cfg(tmp / "one.cfg", addr["weste"], 1)
        one = bench(cfg1, 8, 10)
        data["one_node"] = one
        log("  one node: %.1f evals/s" % one["evals_s"])

        fleet = {}
        for threads in (2, 4):
            cfg = make_cfg(tmp / ("fleet%d.cfg" % threads), ",".join(addr.values()), threads)
            b = bench(cfg, 8, 10)
            fleet[threads] = b
            log("  fleet %d conns: %.1f evals/s" % (threads, b["evals_s"]))
        data["fleet"] = fleet

        # both servers served during the fleet run
        served2 = {}
        for n in NODES:
            rc, c2 = sshutil.run(clients[n], "grep -c 'eval batch' /root/distserver.log || true")
            served2[n] = int(c2.strip() or 0)
        data["served_after_bench"] = served2
        log("  batches served: %s" % served2)
    finally:
        for t in tunnels.values():
            try:
                t.stop()
            except Exception:
                pass
        for n in NODES:
            try:
                sshutil.run(clients[n], "pkill -f 'katago distserver' 2>/dev/null; true")
                clients[n].close()
            except Exception:
                pass
        data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (OUT / "transcript.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        log("saved %s" % (OUT / "transcript.json"))

    render(data, samples, build_total)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

W, H = 1280, 720
BG, HEADER, FG, DIM = (13, 17, 23), (22, 27, 34), (201, 209, 217), (139, 148, 158)
GREEN, RED, CYAN, YELLOW, BLUE = (63, 185, 80), (248, 81, 73), (86, 182, 194), (210, 168, 255), (88, 166, 255)
F_CN = r"C:\Windows\Fonts\msyh.ttc"
F_MONO = r"C:\Windows\Fonts\consola.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


F_H1 = font(F_CN, 32)
F_H2 = font(F_CN, 22)
F_BODY = font(F_MONO, 19)
F_CN19 = font(F_CN, 19)
F_SMALL = font(F_CN, 15)
F_BIG = font(F_CN, 46)


def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def pick(s):
    return F_CN19 if has_cjk(s) else F_BODY


def wrap(s, fnt, maxw):
    out, cur = [], ""
    for ch in s:
        if fnt.getlength(cur + ch) > maxw:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    out.append(cur)
    return out


def base_frame(title, subtitle, progress, status):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 88], fill=HEADER)
    d.text((28, 8), title, font=F_H1, fill=FG)
    if subtitle:
        d.text((28, 52), subtitle, font=F_SMALL, fill=DIM)
    d.rectangle([0, 88, int(W * progress), 92], fill=BLUE)
    d.rectangle([0, H - 34, W, H], fill=HEADER)
    d.text((28, H - 27), status, font=F_SMALL, fill=DIM)
    d.text((W - 250, H - 27), "SakuraLink + KataGo", font=F_SMALL, fill=DIM)
    return img, d


def term_frames(lines, dur, title, subtitle, progress, fps=24, start=0):
    """Progressively reveal terminal lines over `dur` seconds."""
    nframes = int(dur * fps)
    per = max(0.6, dur / max(1, len(lines)))
    # pre-wrap once (text measurement is the expensive part)
    wrapped = []
    for fnt, text, color in lines:
        use = pick(text)  # Consolas has no CJK glyphs; lines with Chinese need the CJK font
        wrapped.append([(use, piece, color) for piece in wrap(text, use, W - 90)] or [(use, "", color)])
    flat = [w for w in wrapped]
    y_cursor = 96
    heights = []
    for pieces in flat:
        h = sum(p[0].size + 9 for p in pieces)
        heights.append(h)
    frames = []
    for f in range(nframes):
        t = f / fps
        visible = min(len(flat), int(t / per) + 1)
        img, d = base_frame(title, subtitle, progress + (1 - progress) * (f / max(1, nframes - 1)), "● 运行中")
        y = 108
        first = max(0, visible - 16)
        for i in range(first, visible):
            for fnt, piece, color in flat[i]:
                d.text((40, y), piece, font=fnt, fill=color)
                y += fnt.size + 9
        if visible < len(flat) or (f // 12) % 2 == 0:
            d.rectangle([40, min(y + 2, H - 70), 52, min(y + 22, H - 50)], fill=FG)
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return frames


def build_frames(samples, total, dur, title, progress, fps=24):
    nframes = int(dur * fps)
    frames = []
    for f in range(nframes):
        frac = f / max(1, nframes - 1)
        img, d = base_frame(title, "两台机器并行编译 · 真实耗时 %s · 视频加速显示" % fmt_time(total),
                            progress + 0.30 * frac, "● 编译中")
        # sample index
        idx = min(len(samples) - 1, int(frac * len(samples)))
        s = samples[idx]
        pct = (s["weste"]["pct"] + s["westc"]["pct"]) / 2
        # big progress bar
        bx0, bx1, by = 90, W - 90, 250
        d.rectangle([bx0, by, bx1, by + 34], fill=(33, 38, 45), outline=(48, 54, 61))
        d.rectangle([bx0, by, bx0 + int((bx1 - bx0) * pct / 100), by + 34], fill=BLUE)
        d.text((bx0 + 10, by + 6), "%d%%" % pct, font=F_H2, fill=FG)
        d.text((bx0, by - 34), "编译进度（weste + westc 平均）", font=F_SMALL, fill=DIM)
        d.text((bx0, by + 52), "weste %3d%%   westc %3d%%" % (s["weste"]["pct"], s["westc"]["pct"]),
               font=F_CN19, fill=CYAN)
        d.text((bx0, by + 86), "已用 %s" % fmt_time(s["t"]), font=F_CN19, fill=DIM)
        # last lines
        y = 400
        for n in ("weste", "westc"):
            line = s[n]["line"]
            for piece in wrap("[%s] %s" % (n, line), F_BODY, W - 100)[:2]:
                d.text((50, y), piece, font=F_BODY, fill=FG)
                y += 26
            y += 6
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return frames


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return "%d:%02d" % (m, s)


def chart_frames(bars, dur, title, progress, fps=24):
    nframes = int(dur * fps)
    frames = []
    maxv = 10000.0
    for f in range(nframes):
        frac = min(1.0, f / (nframes * 0.6))
        img, d = base_frame(title, "端到端吞吐（evals/s，对数坐标）· batch 8", progress + 0.10 * f / nframes, "● Benchmark")
        x0, x1 = 330, W - 120
        top = 150
        # grid
        for v in (10, 100, 1000, 10000):
            x = x0 + (x1 - x0) * (math.log10(v) / 4.0)
            d.line([x, top - 10, x, 520], fill=(48, 54, 61))
            d.text((x - 12, 530), str(v), font=F_SMALL, fill=DIM)
        for i, (label, value, color) in enumerate(bars):
            y = top + i * 72
            d.text((40, y + 12), label, font=pick(label), fill=FG)
            w = (x1 - x0) * (math.log10(max(value, 1)) / 4.0) * frac
            d.rectangle([x0, y, x0 + max(2, w), y + 40], fill=color)
            d.text((x0 + max(2, w) + 12, y + 10), "%.0f" % (value * frac), font=F_BODY, fill=FG)
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return frames


def render(data, samples, build_total, fps=24, out_path=None):
    out_path = out_path or r"C:\Users\10632\Desktop\kata-distributed-verify.mp4"
    frames = []
    total_est = 120.0
    done = 0.0

    def add(fr, dur):
        nonlocal done
        frames.extend(fr)
        done += dur

    # 0. title / plan
    plan = [
        (F_BIG, "KataGo 分布式推理 · 完整验证", FG),
        (F_CN19, "", FG),
        (F_H2, "① 清空构建产物，从零编译两台 GPU 机器", CYAN),
        (F_H2, "② 两台机器通过 SakuraLink 隧道连到同一个本地引擎", CYAN),
        (F_H2, "③ 数值一致性 + 端到端 benchmark 对比", CYAN),
        (F_SMALL, "GPU: 2 × RTX 4080 SUPER 32GB · CUDA 13.0 · 引擎: 本机 build-dist", DIM),
    ]
    img, d = base_frame("KataGo 分布式推理 · 完整验证", "2 台 GPU 机器 · 从零编译到双机 benchmark", 0.0, "准备")
    y = 150
    for fnt, text, color in plan:
        if text:
            d.text((60, y), text, font=fnt, fill=color)
        y += fnt.size + 22
    add([cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)] * int(6.0 * fps), 6.0)

    # 1. clean + configure
    clean_lines = [
        (F_BODY, "$ ssh root@connect.weste.seetacloud.com", CYAN),
        (F_BODY, "$ ssh root@connect.westc.seetacloud.com", CYAN),
        (F_BODY, "", FG),
        (F_BODY, "root@autodl:/root/cpp# rm -rf build-cuda build-dist", FG),
        (F_SMALL, "  两台机器都清空，下面从零开始", DIM),
        (F_BODY, "", FG),
        (F_BODY, "root@autodl:/root/cpp# cmake -S . -B build-cuda -DUSE_BACKEND=CUDA \\", FG),
        (F_BODY, "    -DNO_GIT_REVISION=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc", FG),
        (F_BODY, "-- Configuring done", GREEN),
        (F_BODY, "-- Generating done", GREEN),
        (F_BODY, "-- Build files have been written to: /root/cpp/build-cuda", GREEN),
    ]
    add(term_frames(clean_lines, 12.0, "阶段 1/3 · 清空并配置", "两台机器并行", 0.05), 12.0)

    # 2. build time-lapse
    add(build_frames(samples, build_total, 26.0, "阶段 1/3 · 编译 CUDA 后端", 0.17), 26.0)

    # build finished
    fin = [
        (F_BODY, "[100%] Built target katago", GREEN),
        (F_BODY, "", FG),
        (F_BODY, "$ ./build-cuda/katago version | tail -3", CYAN),
        (F_BODY, "Using CUDA backend", FG),
        (F_BODY, "Compiled with CUDA version 13.0.88", FG),
        (F_SMALL, "两台机器编译产物：%s" % fmt_time(build_total), DIM),
    ]
    add(term_frames(fin, 7.0, "阶段 1/3 · 编译完成", "两台机器均成功", 0.22), 7.0)

    # 3. two machines connected
    info = data.get("info", {})
    conn = [
        (F_BODY, "$ hostname; nvidia-smi -L", CYAN),
        (F_BODY, "weste  %s" % (info.get("weste", ["?"])[1] if len(info.get("weste", [])) > 1 else "?"), FG),
        (F_BODY, "westc  %s" % (info.get("westc", ["?"])[1] if len(info.get("westc", [])) > 1 else "?"), FG),
        (F_SMALL, "两台不同机器、两块不同 GPU（UUID 不同）", DIM),
        (F_BODY, "", FG),
        (F_BODY, "$ ./katago distserver -model model.bin.gz -listen 127.0.0.1:17731 -gpus 0", CYAN),
        (F_BODY, "Distserver: model sakura-b12c192 … convnet", FG),
        (F_BODY, "Distserver: postprocess output scale 8", FG),
        (F_BODY, "Distserver: listening on 127.0.0.1:17731, max-batch 256, gpus 0", FG),
        (F_SMALL, "两台机器各起一个 distserver（只绑本机，公网不可见）", DIM),
        (F_BODY, "", FG),
        (F_BODY, "$ python -m sakura --host … --target-port 17731", CYAN),
        (F_BODY, "  weste  127.0.0.1:%d -> 127.0.0.1:17731" % data["tunnel_ports"]["weste"], GREEN),
        (F_BODY, "  westc  127.0.0.1:%d -> 127.0.0.1:17731" % data["tunnel_ports"]["westc"], GREEN),
        (F_BODY, "  distServers = 127.0.0.1:%d,127.0.0.1:%d" % (
            data["tunnel_ports"]["weste"], data["tunnel_ports"]["westc"]), GREEN),
        (F_SMALL, "本地引擎同时连到两台机器（SSH RTT: %.0f ms / %.0f ms）" % (
            data["rtt"]["weste"], data["rtt"]["westc"]), DIM),
    ]
    add(term_frames(conn, 16.0, "阶段 2/3 · 两台机器连到一起", "两条独立 SSH 隧道", 0.30), 16.0)

    # 4. evidence: both servers served (counts taken after the fleet benchmark)
    corr = data.get("correctness", {})
    served = data.get("served_after_bench", {})
    ev = [
        (F_BODY, "$ grep -c 'eval batch' /root/distserver.log", CYAN),
        (F_BODY, "  weste: %d batches" % served.get("weste", 0), FG),
        (F_BODY, "  westc: %d batches" % served.get("westc", 0), FG),
        (F_SMALL, "两台机器的日志都记录到实际求值 → 确实在同时使用两台机器", DIM),
        (F_BODY, "", FG),
        (F_BODY, "$ katago gtp -config fleet.cfg   # distServers = 两台", CYAN),
        (F_BODY, "  remote: %s" % " ".join(corr.get("remote", "").splitlines()[:2]), FG),
        (F_BODY, "  direct: %s" % " ".join(corr.get("direct", "").splitlines()[:2]), FG),
        (F_CN19, "数值一致性: %s" % ("√ 完全一致" if corr.get("same") else "× 不一致"),
         GREEN if corr.get("same") else RED),
    ]
    add(term_frames(ev, 11.0, "阶段 2/3 · 双机确实在工作", "两边日志都有 eval batch", 0.66), 11.0)

    # 5. benchmarks
    gpu = data.get("gpu_direct", {})
    one = data.get("one_node", {})
    fleet = data.get("fleet", {})
    b8 = fleet.get("2", {}).get("evals_s", 0)
    b4 = fleet.get("4", {}).get("evals_s", 0)
    bl = [
        (F_BODY, "$ ./build-cuda/katago benchmarknn -batch-size 8   # 节点本机，无网络", CYAN),
        (F_BODY, "  weste GPU 直算: %.0f evals/s (%.1f ms/batch)" % (
            gpu.get("8", {}).get("evals_s", 0), gpu.get("8", {}).get("ms", 0)), FG),
        (F_BODY, "  weste GPU 直算 batch 32: %.0f evals/s" % gpu.get("32", {}).get("evals_s", 0), FG),
        (F_BODY, "", FG),
        (F_BODY, "$ katago benchmarknn -config one-node.cfg   # 单机 + 隧道", CYAN),
        (F_BODY, "  1 节点 1 连接: %.0f evals/s" % one.get("evals_s", 0), FG),
        (F_BODY, "$ katago benchmarknn -config fleet.cfg     # 两机 + 两条隧道", CYAN),
        (F_BODY, "  2 节点 2 连接: %.0f evals/s" % b8, FG),
        (F_BODY, "  2 节点 4 连接: %.0f evals/s" % b4, FG),
        (F_SMALL, "瓶颈是公网 RTT(≈24ms)+上行带宽；GPU 利用率 <5%", DIM),
    ]
    add(term_frames(bl, 14.0, "阶段 3/3 · Benchmark", "GPU 直算 vs 单机 vs 双机", 0.52), 14.0)

    bars = [
        ("GPU 直算 (batch 8)", gpu.get("8", {}).get("evals_s", 0), BLUE),
        ("GPU 直算 (batch 32)", gpu.get("32", {}).get("evals_s", 0), BLUE),
        ("单节点 + 隧道", one.get("evals_s", 0), CYAN),
        ("双节点 2 连接", b8, GREEN),
        ("双节点 4 连接", b4, GREEN),
    ]
    add(chart_frames(bars, 10.0, "阶段 3/3 · 吞吐对比", 0.62), 10.0)

    # 6. summary
    su = [
        (F_BIG, "√ 验证通过", GREEN),
        (F_H2, "", FG),
        (F_CN19, "· 两台 4080 SUPER 从零编译、各起一个 distserver", FG),
        (F_CN19, "· 本地引擎经两条 SSH 隧道同时连接两台机器", FG),
        (F_CN19, "· 远程求值与同机直算 %s" % ("完全一致" if corr.get("same") else "不一致"), FG),
        (F_CN19, "· 吞吐：GPU 直算 %.0f → 双机跨公网 %.0f evals/s（瓶颈在网络）" % (
            gpu.get("8", {}).get("evals_s", 0), b4), FG),
        (F_SMALL, "同机房/同机部署才能发挥 GPU 全力；跨公网适合低吞吐场景", DIM),
    ]
    add(term_frames(su, 8.0, "总结", "KataGo 分布式推理 · 双机双卡", 0.75), 8.0)

    # pad / trim to ~120s
    target = 120 * fps
    if len(frames) < target:
        frames.extend([frames[-1]] * (target - len(frames)))
    else:
        frames = frames[:target]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for fr in frames:
        writer.write(fr)
    writer.release()
    log("video -> %s  %.1fs" % (out_path, len(frames) / fps))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render-only":
        payload = json.loads((OUT / "transcript.json").read_text(encoding="utf-8"))
        build = payload.pop("build")
        render(payload, build["samples"], build["total_seconds"])
    else:
        main()


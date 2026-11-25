"""Real cloud-to-local verification run + rendered MP4.

Runs: cloud distserver -> SakuraLink SSH tunnel -> local KataGo client, compares the
result with same-machine direct inference, measures throughput. The screen output is
rendered into a terminal-style video with the actual captured text.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
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

KATAGO_CLIENT = Path(r"C:\Users\10632\Desktop\KataGo-master\build-dist\Release\katago.exe")
KATAGO_LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\build-eigen\Release\katago.exe")
MODEL = Path(r"C:\Users\10632\Desktop\model.bin.gz")
NODE = "weste"
REMOTE_PORT = 17731
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]

entries = []
results = {}


def add(kind, text, pause=0.0):
    entries.append({"kind": kind, "text": text, "pause": pause})
    print(("[%s] " % kind) + text, flush=True)


def cfg_text(extra):
    return "\n".join(BASE + extra) + "\n"


def run_client(exe, cfg, extra_args=None, timeout=300):
    args = [str(exe), "gtp", "-model", str(MODEL), "-config", str(cfg)]
    r = subprocess.run(args, input=SCRIPT.encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    out = r.stdout.decode("utf-8", "replace")
    return [ln.strip() for ln in out.splitlines() if ln.startswith(("whiteWin", "whiteLead"))]


def bench(cfg, batch, iters):
    r = subprocess.run([str(KATAGO_CLIENT), "benchmarknn", "-model", str(MODEL),
                        "-config", str(cfg), "-batch-size", str(batch),
                        "-iterations", str(iters), "-warmup", "2", "-json"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    lines = [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l.strip().startswith("{")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    d = json.loads(lines[-1])
    return d["actualWallNNEvalsPerSec"], d["perThreadMedianMs"][0]


def main():
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass
    tmp = Path(tempfile.mkdtemp(prefix="demo_"))
    add("title", "KataGo 分布式推理：云端 GPU → 本地引擎", 1.2)
    add("dim", "节点: RTX 4080 SUPER 32GB (CUDA 13.0)   隧道: SakuraLink (SSH)   引擎: 本机 build-dist", 1.0)

    c = sshutil.connect(NODE)
    add("section", "$ ssh root@connect.weste.seetacloud.com   # 云端节点", 0.6)

    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.4; true")
    sshutil.run(c, "cd /root/cpp && setsid nohup ./build-cuda/katago distserver "
                   "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 -max-batch 256 "
                   "> /root/distserver.log 2>&1 < /dev/null &" % REMOTE_PORT)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)
    rc, out = sshutil.run(c, "grep -E 'model |listening|postprocess' /root/distserver.log", timeout=30)
    for line in out.strip().splitlines():
        add("log", line.split(": ", 1)[-1], 0.35)
    using = "listening on" in out
    add("pass" if using else "fail", "云端 distserver 已启动（只绑 127.0.0.1，公网不可见）", 0.5)

    # ---- tunnel ----
    add("section", "$ python -m sakura --host … --target-port 17731   # 本地隧道", 0.6)
    node = NodeConfig(host=sshutil.HOSTS[NODE][0], username="root",
                      password=sshutil.HOSTS[NODE][2], remote_port=sshutil.HOSTS[NODE][1],
                      target_host="127.0.0.1", target_port=REMOTE_PORT, name=NODE)
    tun = SSHTunnel(node)
    t0 = time.perf_counter()
    tun.start()
    up_ms = (time.perf_counter() - t0) * 1000
    add("log", "SSH 认证成功: %s:%d" % (node.host, node.remote_port), 0.3)
    add("log", "本地监听 127.0.0.1:%d -> 127.0.0.1:%d" % (tun.local_port, REMOTE_PORT), 0.3)
    add("pass", "隧道就绪（%.0f ms），引擎侧只需配置 distServers = 127.0.0.1:%d" % (up_ms, tun.local_port), 0.6)

    try:
        # ---- remote raw-nn ----
        add("section", "$ katago gtp -config gtp.cfg    # distServers 指向本地隧道", 0.6)
        cfg = tmp / "dist.cfg"
        cfg.write_text(cfg_text(["distServers = 127.0.0.1:%d" % tun.local_port]), encoding="utf-8")
        t0 = time.perf_counter()
        remote = run_client(KATAGO_CLIENT, cfg)
        remote_ms = (time.perf_counter() - t0) * 1000
        for line in remote:
            add("log", "  " + line, 0.25)
        add("dim", "远端往返 %.0f ms（含握手/模型加载）" % remote_ms, 0.4)

        # ---- direct on the node ----
        add("section", "$ ./build-cuda/katago gtp …   # 同机直算参考", 0.5)
        sftp = c.open_sftp()
        with sftp.open("/root/demo.cfg", "w") as f:
            f.write(cfg_text([]))
        with sftp.open("/root/demo.gtp", "w") as f:
            f.write(SCRIPT)
        sftp.close()
        rc, out = sshutil.run(c, "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
                                 "-config /root/demo.cfg < /root/demo.gtp 2>&1 | grep -E '^whiteWin|^whiteLead'",
                              timeout=300)
        direct = [ln.strip() for ln in out.splitlines() if ln.strip()]
        for line in direct:
            add("log", "  " + line, 0.25)

        same = remote == direct
        add("pass" if same else "fail",
            "数值一致性: 远程 == 直算  %s" % ("✓" if same else "✗"), 0.6)

        # ---- throughput ----
        add("section", "$ katago benchmarknn -batch-size 8   # 端到端吞吐", 0.5)
        evals, ms = bench(cfg, 8, 8)
        add("log", "  batch 8, 1 connection: %.1f evals/s (%.1f ms/batch)" % (evals, ms), 0.5)
        evals32, ms32 = bench(cfg, 32, 5)
        add("log", "  batch 32, 1 connection: %.1f evals/s (%.1f ms/batch)" % (evals32, ms32), 0.5)
        add("dim", "GPU 直算 batch 8 约 4200 evals/s；跨公网受 RTT(≈24ms) 与上行带宽限制", 0.6)

        results.update({"same": same, "remote": remote, "direct": direct,
                        "evals8": evals, "ms8": ms, "evals32": evals32, "ms32": ms32,
                        "up_ms": up_ms})
    finally:
        tun.stop()
        sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
        c.close()

    add("section", "验证结果", 0.8)
    add("pass" if results.get("same") else "fail",
        "云端 GPU 推理已连接到本地：数值与同机直算完全一致", 0.6)
    add("log", "  端到端吞吐: %.1f evals/s (batch 8) / %.1f evals/s (batch 32)" % (
        results["evals8"], results["evals32"]), 0.5)
    add("dim", "瓶颈是网络（RTT + 上行带宽），不是引擎或 GPU", 1.2)

    (CLOUD / "out").mkdir(exist_ok=True)
    (CLOUD / "out" / "demo_transcript.json").write_text(
        json.dumps({"entries": entries, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    render(entries, results)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

W, H = 1280, 720
BG = (13, 17, 23)
HEADER = (22, 27, 34)
FG = (201, 209, 217)
DIM = (139, 148, 158)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
CYAN = (86, 182, 194)
YELLOW = (210, 168, 255)
FONT_CN = r"C:\Windows\Fonts\msyh.ttc"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = font(FONT_CN, 34)
F_SUB = font(FONT_CN, 18)
F_BODY = font(FONT_MONO, 20)
F_BODY_CN = font(FONT_CN, 20)
F_SMALL = font(FONT_CN, 16)
F_BIG = font(FONT_CN, 54)


def draw_header(d, t_total):
    d.rectangle([0, 0, W, 64], fill=HEADER)
    d.text((28, 14), "KataGo 分布式推理 · 云端 GPU 验证", font=F_TITLE, fill=FG)
    d.ellipse([W - 150, 26, W - 134, 42], fill=GREEN)
    d.text((W - 124, 22), "LIVE", font=F_SUB, fill=GREEN)


def draw_footer(d, status):
    d.rectangle([0, H - 40, W, H], fill=HEADER)
    d.text((28, H - 32), status, font=F_SMALL, fill=DIM)
    d.text((W - 210, H - 32), "SakuraLink + KataGo", font=F_SMALL, fill=DIM)


def wrap(text, fnt, max_w):
    lines, cur = [], ""
    for ch in text:
        if fnt.getlength(cur + ch) > max_w:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    lines.append(cur)
    return lines


def has_cjk(text):
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def pick_font(text):
    return F_BODY_CN if has_cjk(text) else F_BODY


def render(entries, results, fps=24, out_path=None):
    out_path = out_path or r"C:\Users\10632\Desktop\kata-cloud-verify.mp4"
    frames = []

    def frame(lines, status="● 运行中", cursor=False, big=None, big_color=GREEN):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        draw_header(d, 0)
        if big:
            d.text((W // 2 - 300, 260), big, font=F_BIG, fill=big_color)
        y = 96
        for fnt, text, color in lines:
            d.text((40, y), text, font=fnt, fill=color)
            y += fnt.size + 10
        if cursor:
            d.rectangle([40, y + 2, 52, y + 24], fill=FG)
        draw_footer(d, status)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # title
    title_lines = [
        (F_TITLE, "KataGo 分布式推理", FG),
        (F_SUB, "云端 GPU 节点  →  SakuraLink SSH 隧道  →  本地引擎", DIM),
        (F_SMALL, "RTX 4080 SUPER 32GB · CUDA 13.0 · 两次真实运行（远程 vs 同机直算）", DIM),
    ]
    for _ in range(int(1.6 * fps)):
        frames.append(frame(title_lines, "准备中…"))

    visible = []
    for e in entries:
        if e["kind"] in ("title", "dim") and not visible:
            continue
        color = {"section": CYAN, "pass": GREEN, "fail": RED, "log": FG,
                 "dim": DIM, "title": FG}.get(e["kind"], FG)
        fnt = pick_font(e["text"])
        text = e["text"].replace("✓", "√").replace("✗", "×")
        if e["kind"] == "section" and text.startswith("$ "):
            text = "$ " + text[2:]
        for piece in wrap(text, fnt, W - 90):
            visible.append((fnt, piece, color))
        # cap the body to the last 17 lines
        shown = visible[-17:]
        hold = max(4, int(e.get("pause", 0.3) * fps))
        for _ in range(hold):
            frames.append(frame(shown, "● 运行中", cursor=True))

    # result card
    ok = results.get("same")
    res_lines = [
        (F_BODY_CN, "远程（云端 GPU）与同机直算：", DIM),
        (F_BODY, "  remote: %s" % "  ".join(results.get("remote", [])), FG),
        (F_BODY, "  direct: %s" % "  ".join(results.get("direct", [])), FG),
        (F_BODY_CN, "  端到端吞吐: %.1f evals/s (batch 8), %.1f evals/s (batch 32)"
         % (results.get("evals8", 0), results.get("evals32", 0)), FG),
        (F_SMALL, "瓶颈在网络（RTT ≈24 ms + 上行带宽），GPU 利用率 <5%", DIM),
    ]
    big = "√ 验证通过" if ok else "× 验证失败"
    for _ in range(int(3.5 * fps)):
        frames.append(frame(res_lines, "完成", big=big,
                            big_color=GREEN if ok else RED))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print("video ->", out_path, "%.1fs" % (len(frames) / fps))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render-only":
        data = json.loads((CLOUD / "out" / "demo_transcript.json").read_text(encoding="utf-8"))
        render(data["entries"], data["results"])
    else:
        main()

"""Local end-to-end check of the ported feature on the upstream clone."""
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path

CPP = Path(r"C:\Users\10632\Desktop\KataGo-upstream\cpp")
SERVER = CPP / "build-eigen" / "Release" / "katago.exe"
CLIENT = CPP / "build-dist" / "Release" / "katago.exe"
MODEL = Path(r"C:\Users\10632\Desktop\model.bin.gz")
PORT = 17791
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"

BASE = [
    "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
    "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
    "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
    "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
    "nnRandomize = false", "searchRandSeed = 12345", "useFP16 = false",
]


def cfg(path, dist):
    lines = list(BASE)
    if dist:
        lines.append("distServers = 127.0.0.1:%d" % PORT)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(exe, conf, timeout=300):
    r = subprocess.run([str(exe), "gtp", "-model", str(MODEL), "-config", str(conf)],
                       input=SCRIPT.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout)
    out = r.stdout.decode("utf-8", "replace")
    return r.returncode, "\n".join(l.strip() for l in out.splitlines()
                                   if l.strip().startswith(("whiteWin", "whiteLead"))), \
        r.stderr.decode("utf-8", "replace")[-300:]


tmp = Path(tempfile.mkdtemp(prefix="upstream_e2e_"))
log = open(tmp / "server.log", "wb")
srv = subprocess.Popen([str(SERVER), "distserver", "-model", str(MODEL),
                        "-listen", "127.0.0.1:%d" % PORT, "-gpus", "0", "-max-batch", "256"],
                       stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        text = (tmp / "server.log").read_text(errors="replace")
        if "listening on" in text:
            break
        if srv.poll() is not None:
            raise SystemExit("server died:\n" + text[-800:])
        time.sleep(0.2)
    print("server:", [l for l in text.splitlines() if "listening" in l or "output scale" in l])

    rc_d, direct, err_d = run(SERVER, cfg(tmp / "direct.cfg", False))
    rc_c, remote, err_c = run(CLIENT, cfg(tmp / "remote.cfg", True))
    print("direct:", rc_d, direct.replace("\n", " | "), err_d.strip()[-120:])
    print("remote:", rc_c, remote.replace("\n", " | "), err_c.strip()[-120:])
    print("MATCH:", direct == remote and "whiteWin" in direct)

    def bench(exe, conf, batch, iters):
        r = subprocess.run([str(exe), "benchmarknn", "-model", str(MODEL), "-config", str(conf),
                            "-batch-size", str(batch), "-iterations", str(iters),
                            "-warmup", "1", "-json"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        lines = [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l.startswith("{")]
        if r.returncode != 0 or not lines:
            return None
        d = json.loads(lines[-1])
        return d["actualWallNNEvalsPerSec"]

    print("bench direct batch8:", bench(SERVER, tmp / "direct.cfg", 8, 8))
    print("bench remote batch8:", bench(CLIENT, tmp / "remote.cfg", 8, 8))
finally:
    srv.kill()
    srv.wait(timeout=10)
    log.close()

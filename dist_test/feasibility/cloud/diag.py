"""Diagnose the raw-NN mismatch: local Eigen vs node-local CUDA vs tunneled client.

The earlier "byte-identical" check compared an FP32 CPU backend with an FP16 GPU
backend, which cannot be byte-equal. The right equivalence test for the
distributed path is: the tunneled client's output must equal what the *same*
node produces locally (same backend, same batch shape).
"""
import sys
import tempfile
import time
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, r"C:\Users\10632\Desktop\SakuraLink")
import sshutil  # noqa: E402
from sakura.config import NodeConfig  # noqa: E402
from sakura.tunnel import SSHTunnel  # noqa: E402
from cloud_test import (KATAGO_CLIENT, KATAGO_LOCAL, MODEL, REMOTE_DS_PORT,  # noqa: E402
                        make_cfg, gtp, raw_text)

SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b all\nquit\n"


def parse_blocks(text):
    blocks, cur = {}, None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "symmetry":
            cur = int(parts[1])
            blocks[cur] = {}
        elif cur is not None and len(parts) >= 2:
            try:
                blocks[cur][parts[0]] = float(parts[1])
            except ValueError:
                pass
    return blocks


def main():
    tmp = Path(tempfile.mkdtemp(prefix="diag_"))
    node = "weste"
    host, port, pw = sshutil.HOSTS[node]
    c = sshutil.connect(node)

    # node-local CUDA reference (same shape: 1 thread, max batch 1)
    cfg_text = "\n".join([
        "logToStderr = false",
        "logAllGTPCommunication = false",
        "logSearchInfo = false",
        "rules = tromp-taylor",
        "allowResignation = false",
        "resignThreshold = -0.90",
        "resignConsecTurns = 3",
        "ponderingEnabled = false",
        "maxVisits = 1",
        "numSearchThreads = 1",
        "nnMaxBatchSize = 1",
        "numNNServerThreadsPerModel = 1",
        "nnRandomize = false",
        "searchRandSeed = 12345",
    ]) + "\n"
    sftp = c.open_sftp()
    with sftp.open("/root/diag.cfg", "w") as f:
        f.write(cfg_text)
    sftp.close()

    # restart a distserver with -max-batch large, but client declares 1
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; sleep 0.5; true")
    sshutil.run(c, "cd /root/cpp && setsid nohup ./build-cuda/katago distserver "
                   "-model /root/model.bin.gz -listen 127.0.0.1:%d -gpus 0 "
                   "-max-batch 256 -boardsize 19 -timeout 600 "
                   "> /root/distserver.log 2>&1 < /dev/null &" % REMOTE_DS_PORT)
    for _ in range(60):
        rc, out = sshutil.run(c, "grep -c 'listening on' /root/distserver.log 2>/dev/null || true")
        if out.strip().isdigit() and int(out.strip()) > 0:
            break
        time.sleep(0.5)

    # node-local CUDA run (feed the GTP script through a file; paramiko exec has no stdin)
    sftp = c.open_sftp()
    with sftp.open("/root/diag.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(
        c, "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
           "-config /root/diag.cfg < /root/diag.gtp 2>/dev/null", timeout=300)
    cuda_text = raw_text(out.encode())

    tun = SSHTunnel(NodeConfig(host=host, username="root", password=pw,
                               remote_port=port, target_host="127.0.0.1",
                               target_port=REMOTE_DS_PORT))
    tun.start()
    try:
        cfg_t = make_cfg(tmp / "tun.cfg", "127.0.0.1:%d" % tun.local_port, 1, batch=1)
        r = gtp(KATAGO_CLIENT, cfg_t, SCRIPT, timeout=300)
        tun_text = raw_text(r.stdout)
    finally:
        tun.stop()

    cfg_e = make_cfg(tmp / "eigen.cfg", None, 1, batch=1,
                     extra={"numEigenThreadsPerModel": 1})
    r = gtp(KATAGO_LOCAL, cfg_e, SCRIPT, timeout=300)
    eigen_text = raw_text(r.stdout)

    blocks = {"cuda_local": parse_blocks(cuda_text),
              "tunnel": parse_blocks(tun_text),
              "eigen": parse_blocks(eigen_text)}

    print("cuda_local == tunnel (byte):", cuda_text == tun_text)
    print("cuda_local == eigen  (byte):", cuda_text == eigen_text)
    print()
    print("%-4s %14s %14s %14s" % ("sym", "cuda_local", "tunnel", "eigen"))
    for sym in sorted(blocks["cuda_local"]):
        a = blocks["cuda_local"][sym].get("whiteWin")
        b = blocks["tunnel"][sym].get("whiteWin")
        e = blocks["eigen"][sym].get("whiteWin")
        print("%-4d %14.8f %14.8f %14.8f" % (sym, a, b, e))
    diffs = [abs(blocks["tunnel"][s]["whiteWin"] - blocks["eigen"][s]["whiteWin"])
             for s in blocks["tunnel"] if s in blocks["eigen"]]
    print("\nmax |tunnel - eigen| whiteWin = %.3g" % max(diffs))

    rc, out = sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()


if __name__ == "__main__":
    main()

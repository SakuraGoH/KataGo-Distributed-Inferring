"""Loopback root-cause test on weste: node dist client vs node direct engine."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil
from cloud_test import REMOTE_DS_PORT, start_distserver  # noqa: E402

NODE = "weste"
SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"


def cfg_text(dist, layout=None, fp16=None):
    lines = [
        "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
        "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
        "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
        "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
        "nnRandomize = false", "searchRandSeed = 12345",
    ]
    if dist:
        lines.append("distServers = 127.0.0.1:%d" % REMOTE_DS_PORT)
    if layout is not None:
        lines.append("inputsUseNHWC = %s" % layout)
    if fp16 is not None:
        lines.append("useFP16 = %s" % fp16)
    return "\n".join(lines) + "\n"


def run_case(c, name, cfg, exe):
    sftp = c.open_sftp()
    with sftp.open("/root/rc.cfg", "w") as f:
        f.write(cfg)
    with sftp.open("/root/rc.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, out = sshutil.run(
        c, "cd /root/cpp && %s gtp -model /root/model.bin.gz -config /root/rc.cfg "
           "< /root/rc.gtp 2>&1 | grep -E 'whiteWin|whiteLead|Uncaught|Error'" % exe,
        timeout=300)
    vals = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in ("whiteWin", "whiteLead"):
            vals[p[0]] = p[1]
        elif "Uncaught" in line or "Error" in line:
            vals.setdefault("err", line.strip()[:120])
    print("%-34s %s" % (name, vals), flush=True)


c = sshutil.connect(NODE)
start_distserver(NODE).close()
try:
    run_case(c, "direct cuda fp32 nhwc", cfg_text(False, layout="true", fp16="false"), "./build-cuda/katago")
    run_case(c, "direct cuda fp32 nchw", cfg_text(False, layout="false", fp16="false"), "./build-cuda/katago")
    run_case(c, "loopback dist client (nhwc)", cfg_text(True, layout="true"), "./build-dist/katago")
    run_case(c, "loopback dist client (nchw)", cfg_text(True, layout="false"), "./build-dist/katago")
    rc, out = sshutil.run(c, "grep -E 'connection [0-9]+ from' /root/distserver.log", timeout=30)
    print("\nserver log connections:")
    print("\n".join(out.splitlines()[-6:]))
finally:
    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()

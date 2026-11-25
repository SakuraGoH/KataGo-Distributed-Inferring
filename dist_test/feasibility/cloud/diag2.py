"""Narrow down the raw-NN difference: precision vs layout vs protocol."""
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

SCRIPT = "boardsize 19\nkomi 6.5\nkata-raw-nn b 0\nquit\n"
NODE = "weste"


def base_cfg(extra):
    lines = [
        "logToStderr = false", "logAllGTPCommunication = false", "logSearchInfo = false",
        "rules = tromp-taylor", "allowResignation = false", "resignThreshold = -0.90",
        "resignConsecTurns = 3", "ponderingEnabled = false", "maxVisits = 1",
        "numSearchThreads = 1", "nnMaxBatchSize = 1", "numNNServerThreadsPerModel = 1",
        "nnRandomize = false", "searchRandSeed = 12345",
    ] + ["%s = %s" % kv for kv in extra.items()]
    return "\n".join(lines) + "\n"


def node_raw(c, cfg_text, script):
    sftp = c.open_sftp()
    with sftp.open("/root/d2.cfg", "w") as f:
        f.write(cfg_text)
    with sftp.open("/root/d2.gtp", "w") as f:
        f.write(script)
    sftp.close()
    rc, out = sshutil.run(
        c, "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
           "-config /root/d2.cfg < /root/d2.gtp 2>&1 | grep -E 'whiteWin|whiteLead'",
        timeout=300)
    return out.strip()


def first_value(text, key):
    for line in text.splitlines():
        p = line.strip().split()
        if len(p) >= 2 and p[0] == key:
            return float(p[1])
    return None


def main():
    tmp = Path(tempfile.mkdtemp(prefix="diag2_"))
    host, port, pw = sshutil.HOSTS[NODE]
    c = sshutil.connect(NODE)

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

    tun = SSHTunnel(NodeConfig(host=host, username="root", password=pw,
                               remote_port=port, target_host="127.0.0.1",
                               target_port=REMOTE_DS_PORT))
    tun.start()
    try:
        cases = {}
        cases["node cuda fp16(auto)"] = node_raw(c, base_cfg({}), SCRIPT)
        cases["node cuda fp32"] = node_raw(c, base_cfg({"useFP16": "false"}), SCRIPT)

        for tag, layout in (("tunnel nhwc", None), ("tunnel nchw", "false")):
            extra = {"inputsUseNHWC": layout} if layout else {}
            cfg = make_cfg(tmp / (tag.replace(" ", "_") + ".cfg"),
                           "127.0.0.1:%d" % tun.local_port, 1, batch=1, extra=extra)
            r = gtp(KATAGO_CLIENT, cfg, SCRIPT, timeout=300)
            cases[tag] = raw_text(r.stdout)
    finally:
        tun.stop()

    cfg_e = make_cfg(tmp / "eigen.cfg", None, 1, batch=1,
                     extra={"numEigenThreadsPerModel": 1})
    r = gtp(KATAGO_LOCAL, cfg_e, SCRIPT, timeout=300)
    cases["local eigen fp32"] = raw_text(r.stdout)

    print("%-22s %12s %12s" % ("case", "whiteWin", "whiteLead"))
    for name, text in cases.items():
        print("%-22s %12.8f %12.4f" % (name, first_value(text, "whiteWin") or -1,
                                       first_value(text, "whiteLead") or -1))

    sshutil.run(c, "pkill -f 'katago distserver' 2>/dev/null; true")
    c.close()


if __name__ == "__main__":
    main()

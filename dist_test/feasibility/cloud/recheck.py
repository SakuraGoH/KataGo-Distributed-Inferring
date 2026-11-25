"""Recompute the correctness block of the saved demo transcript, then re-render.

The first demo run compared raw logs (with startup lines) against the remote values, so
the MATCH verdict was wrong. This re-runs just that comparison with filtered output.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"C:\Users\10632\Desktop\SakuraLink")
import sshutil  # noqa: E402
from sakura.config import NodeConfig  # noqa: E402
from sakura.tunnel import SSHTunnel  # noqa: E402
from cloud_test import (KATAGO_CLIENT, MODEL, REMOTE_DS_PORT, gtp, make_cfg)  # noqa: E402
from demo_video2 import OUT, SCRIPT, start_distserver  # noqa: E402

NODES = ("weste", "westc")
payload = json.loads((OUT / "transcript.json").read_text(encoding="utf-8"))

clients = {}
for n in NODES:
    clients[n], _ = start_distserver(n)

tunnels = {}
try:
    for n in NODES:
        node = NodeConfig(host=sshutil.HOSTS[n][0], username="root", password=sshutil.HOSTS[n][2],
                          remote_port=sshutil.HOSTS[n][1], target_host="127.0.0.1",
                          target_port=REMOTE_DS_PORT, name=n)
        t = SSHTunnel(node)
        t.start()
        tunnels[n] = t
    addr = {n: "127.0.0.1:%d" % tunnels[n].local_port for n in NODES}

    tmp = Path(tempfile.mkdtemp(prefix="recheck_"))
    cfg = make_cfg(tmp / "fleet.cfg", ",".join(addr.values()), 2, extra={"useFP16": "false"})
    r = gtp(KATAGO_CLIENT, cfg, SCRIPT)
    out = r.stdout.decode("utf-8", "replace")
    remote = "\n".join(l.strip() for l in out.splitlines()
                       if l.strip().startswith(("whiteWin", "whiteLead")))

    sftp = clients["weste"].open_sftp()
    with sftp.open("/root/rc2.cfg", "w") as f:
        f.write((tmp / "fleet.cfg").read_text(encoding="utf-8")
                .replace("distServers = %s" % ",".join(addr.values()), ""))
    with sftp.open("/root/rc2.gtp", "w") as f:
        f.write(SCRIPT)
    sftp.close()
    rc, direct_out = sshutil.run(clients["weste"],
                                 "cd /root/cpp && ./build-cuda/katago gtp -model /root/model.bin.gz "
                                 "-config /root/rc2.cfg < /root/rc2.gtp 2>&1 | "
                                 "grep -E '^(whiteWin|whiteLead)'", timeout=300)
    direct = "\n".join(l.strip() for l in direct_out.splitlines() if l.strip())

    print("remote:", remote.replace("\n", " | "))
    print("direct:", direct.replace("\n", " | "))
    print("same:", remote == direct and "whiteWin" in remote)

    payload["correctness"] = {"remote": remote, "direct": direct,
                              "same": remote == direct and "whiteWin" in remote}
    (OUT / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
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

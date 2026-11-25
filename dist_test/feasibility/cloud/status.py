import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

for h in ("weste", "westc"):
    c = sshutil.connect(h)
    rc, out = sshutil.run(
        c,
        "echo servers=$(pgrep -fc 'katago distserver' || true); "
        "ls -l /root/cpp/build-cuda/katago /root/model.bin.gz | awk '{print $5, $9}'; "
        "grep -c KATAGO_DUMP_REQ /root/cpp/neuralnet/nneval.cpp /root/cpp/command/distserver.cpp || true",
        timeout=30)
    print("###", h)
    print(out.strip())
    c.close()

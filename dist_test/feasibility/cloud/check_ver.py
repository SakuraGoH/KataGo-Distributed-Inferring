import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

for h in ("weste", "westc"):
    c = sshutil.connect(h)
    rc, out = sshutil.run(
        c,
        "grep -n 'PROTOCOL_VERSION' /root/cpp/neuralnet/distprotocol.h | head -3; "
        "ls -l /root/cpp/neuralnet/distprotocol.h /root/cpp/command/distserver.cpp; "
        "grep -c outputScaleMultiplier /root/cpp/command/distserver.cpp; "
        "ls -l /root/cpp/build-cuda/katago",
        timeout=30)
    print("###", h)
    print(out.strip())
    c.close()

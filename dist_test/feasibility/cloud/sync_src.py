"""Sync the protocol-v2 source files to a node (used for westc, which still had v1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

NODE = sys.argv[1] if len(sys.argv) > 1 else "westc"
CPP = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
FILES = [
    "command/distserver.cpp",
    "neuralnet/distprotocol.h",
    "neuralnet/distprotocol.cpp",
    "neuralnet/distributedbackend.cpp",
]

c = sshutil.connect(NODE)
sftp = c.open_sftp()
for f in FILES:
    sftp.put(str(CPP / f), "/root/cpp/" + f)
    print("uploaded", f)
sftp.close()
rc, out = sshutil.run(c, "grep -n 'PROTOCOL_VERSION' /root/cpp/neuralnet/distprotocol.h | head -2", timeout=30)
print(out.strip())
c.close()

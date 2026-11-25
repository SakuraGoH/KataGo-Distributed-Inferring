import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

c = sshutil.connect("weste")
rc, out = sshutil.run(c, "grep -n 'KATAGO_DUMP_REQ' -B3 -A2 /root/cpp/neuralnet/nneval.cpp", timeout=30)
print("--- nneval patch ---")
print(out)
rc, out = sshutil.run(c, "cd /root/cpp && KATAGO_DUMP_REQ=/root/dump_direct.bin "
                         "./build-cuda/katago gtp -model /root/model.bin.gz -config /root/dr.cfg "
                         "< /root/dr.gtp 2>&1 | tail -20", timeout=300)
print("--- direct run ---")
print(out)
rc, out = sshutil.run(c, "tail -n 20 /root/distserver.log", timeout=30)
print("--- distserver log ---")
print(out)
c.close()

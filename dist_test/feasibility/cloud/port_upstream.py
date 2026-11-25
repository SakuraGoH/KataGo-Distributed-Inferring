"""Port the distributed-inference feature onto the upstream clone (LF-normalized)."""
import shutil
import subprocess
from pathlib import Path

LOCAL = Path(r"C:\Users\10632\Desktop\KataGo-master\cpp")
UP = Path(r"C:\Users\10632\Desktop\KataGo-upstream")

ADDED = [
    "command/distserver.cpp",
    "neuralnet/distprotocol.cpp",
    "neuralnet/distprotocol.h",
    "neuralnet/distributedbackend.cpp",
]
MODIFIED = [
    "CMakeLists.txt",
    "command/benchmarknn.cpp",
    "main.cpp",
    "main.h",
    "program/setup.cpp",
]


def copy_lf(src, dst):
    data = src.read_bytes().replace(b"\r\n", b"\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return len(data)


for rel in ADDED + MODIFIED:
    n = copy_lf(LOCAL / rel, UP / "cpp" / rel)
    print("%-32s %d bytes" % (rel, n))

# docs (sparse checkout does not include docs/ yet)
subprocess.run(["git", "sparse-checkout", "add", "docs"], cwd=str(UP), check=True)
n = copy_lf(LOCAL.parent / "docs" / "DistributedInference.md", UP / "docs" / "DistributedInference.md")
print("docs/DistributedInference.md     %d bytes" % n)

rc = subprocess.run(["git", "status", "--short"], cwd=str(UP), stdout=subprocess.PIPE, text=True)
print(rc.stdout)

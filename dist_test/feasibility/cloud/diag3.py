"""Is the CUDA backend's NCHW input path the source of the mismatch?"""
import sys
from pathlib import Path

CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(CLOUD))
import sshutil  # noqa: E402
from diag2 import base_cfg, node_raw, first_value, SCRIPT  # noqa: E402

NODE = "weste"
c = sshutil.connect(NODE)
cases = {
    "nhwc fp16": base_cfg({}),
    "nchw fp16": base_cfg({"inputsUseNHWC": "false"}),
    "nhwc fp32": base_cfg({"useFP16": "false"}),
    "nchw fp32": base_cfg({"inputsUseNHWC": "false", "useFP16": "false"}),
}
print("%-12s %12s %12s" % ("case", "whiteWin", "whiteLead"))
for name, cfg in cases.items():
    text = node_raw(c, cfg, SCRIPT)
    print("%-12s %12.8f %12.4f" % (name, first_value(text, "whiteWin") or -1,
                                   first_value(text, "whiteLead") or -1))
c.close()

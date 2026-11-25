"""Open a draft PR from SakuraGoH/KataGo to lightvector/KataGo."""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "lightvector"
REPO = "KataGo"
HEAD = "SakuraGoH:feature/distributed-inference"
BASE = "master"
TITLE = "Distributed inference: engine without a GPU, evals served from remote machines"
BODY = Path(r"C:\Users\10632\Desktop\pr_upstream.md").read_text(encoding="utf-8")


def token():
    out = subprocess.run(["git", "credential", "fill"], input=b"protocol=https\nhost=github.com\n\n",
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode()
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no token")


def api(method, url, tok, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + tok)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "dsh")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, (json.loads(body) if body.strip() else {})


tok = token()
code, prs = api("GET", "https://api.github.com/repos/%s/%s/pulls?state=open&head=%s"
                % (OWNER, REPO, HEAD), tok)
if code == 200:
    for pr in prs:
        if pr.get("head", {}).get("label") == HEAD:
            print("PR already open:", pr["html_url"])
            sys.exit(0)

code, res = api("POST", "https://api.github.com/repos/%s/%s/pulls" % (OWNER, REPO), tok, {
    "title": TITLE, "head": HEAD, "base": BASE, "body": BODY, "draft": True,
})
if code in (200, 201):
    print("PR created:", res["html_url"])
else:
    print("FAILED", code, res.get("message"))
    print(json.dumps(res.get("errors", []), ensure_ascii=False))
    sys.exit(1)

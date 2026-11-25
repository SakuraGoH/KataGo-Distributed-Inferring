"""Is a direct TCP connection to the node even possible, and how fast is it
compared with the SSH tunnel?  Measures raw upload/download throughput.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sshutil

NODE = "weste"
PORT = 23456
SIZE = 32 << 20  # 32 MB

SERVER = r'''
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", %d))
s.listen(8)
for _ in range(6):
    c, _ = s.accept()
    try:
        cmd = c.recv(1)
        if cmd == b"U":
            total = 0
            while True:
                d = c.recv(1 << 20)
                if not d:
                    break
                total += len(d)
            print("recv", total, flush=True)
        elif cmd == b"D":
            buf = b"x" * (1 << 20)
            sent = 0
            while sent < %d:
                c.sendall(buf)
                sent += len(buf)
            print("sent", sent, flush=True)
    finally:
        c.close()
''' % (PORT, SIZE)


def measure_upload(host):
    s = socket.create_connection((host, PORT), timeout=8)
    s.sendall(b"U")
    buf = b"x" * (1 << 20)
    t0 = time.perf_counter()
    sent = 0
    while sent < SIZE:
        s.sendall(buf)
        sent += len(buf)
    s.shutdown(socket.SHUT_WR)
    try:
        s.recv(16)
    except OSError:
        pass
    dt = time.perf_counter() - t0
    s.close()
    return SIZE / dt / 1e6, dt


def measure_download(host):
    s = socket.create_connection((host, PORT), timeout=8)
    s.sendall(b"D")
    t0 = time.perf_counter()
    got = 0
    while True:
        d = s.recv(1 << 20)
        if not d:
            break
        got += len(d)
    dt = time.perf_counter() - t0
    s.close()
    return got / dt / 1e6, dt


def main():
    c = sshutil.connect(NODE)
    rc, out = sshutil.run(c, "curl -s --max-time 8 ifconfig.me; echo; "
                             "ip -4 addr show | grep -oP 'inet \\K[\\d.]+' | head -3", timeout=30)
    print("node egress / addrs:\n" + out.strip())

    sftp = c.open_sftp()
    with sftp.open("/root/direct_srv.py", "w") as f:
        f.write(SERVER)
    sftp.close()
    sshutil.run(c, "pkill -f direct_srv.py 2>/dev/null; "
                   "setsid nohup python3 /root/direct_srv.py > /root/direct_srv.log 2>&1 < /dev/null &")
    time.sleep(1.5)

    public_ip = out.strip().splitlines()[0].strip()
    print("\ndirect to %s:%d ..." % (public_ip, PORT))
    try:
        up, dt = measure_upload(public_ip)
        print("  DIRECT upload  : %.1f MB/s (%.2fs)" % (up, dt))
        down, dt = measure_download(public_ip)
        print("  DIRECT download: %.1f MB/s (%.2fs)" % (down, dt))
        direct_ok = True
    except Exception as e:
        print("  direct failed:", type(e).__name__, e)
        direct_ok = False

    # via ssh -L (OpenSSH), same endpoint
    ssh = r"C:\Windows\System32\OpenSSH\ssh.exe"
    key = Path(__file__).parent / "id_ed25519"
    host, port, _ = sshutil.HOSTS[NODE]
    p = subprocess.Popen(
        [ssh, "-N", "-i", str(key), "-p", str(port),
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
         "-o", "ExitOnForwardFailure=yes",
         "-L", "127.0.0.1:%d:127.0.0.1:%d" % (PORT + 1, PORT), "root@" + host],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", PORT + 1), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        up, dt = measure_upload("127.0.0.1")
        print("\n  ssh -L upload  : %.1f MB/s (%.2fs)" % (up, dt))
        down, dt = measure_download("127.0.0.1")
        print("  ssh -L download: %.1f MB/s (%.2fs)" % (down, dt))
    finally:
        p.terminate()

    sshutil.run(c, "pkill -f direct_srv.py 2>/dev/null; rm -f /root/direct_srv.py; true")
    c.close()
    print("\ndirect reachable:", direct_ok)


if __name__ == "__main__":
    main()

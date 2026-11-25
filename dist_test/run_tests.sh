#!/usr/bin/env bash
# NOTE: superseded by dist_test/feasibility/harness.py (see feasibility/REPORT.md).
# Two problems with this script's game checks: nnRandomize=false alone does not make play
# reproducible across processes (searchRandSeed must be fixed too), and it compares two
# separate processes' move lists that are therefore not comparable.
# Distributed-inference test suite v3.
# Correctness is checked by comparing RAW nn output (kata-analyze visits=1, exact match required),
# and by identical deterministic games at numSearchThreads=1 on each server.
# Usage: bash dist_test/run_tests.sh   (from the KataGo-master directory)
set -u
ROOT="/c/Users/10632/Desktop/KataGo-master"
SERVER_EXE="$ROOT/build-eigen/Release/katago.exe"
CLIENT_EXE="$ROOT/build-dist/Release/katago.exe"
MODEL="/c/Users/10632/Desktop/model.bin.gz"
PORT1=17731
PORT2=17732
mkdir -p "$ROOT/dist_test/out"
cd "$ROOT/dist_test"

mkcfg() { # mkcfg <outfile> <numSearchThreads> <maxVisits> <servers-or-empty>
  cp ../cpp/configs/gtp_example.cfg "$1"
  sed -i -e 's/^allowResignation = true/allowResignation = false/' "$1"
  sed -i -e "s/^numSearchThreads = 6/numSearchThreads = $2/" "$1"
  sed -i -e "s/^maxVisits = 500/maxVisits = $3/" "$1"
  if [ -n "$4" ]; then
    printf '\n# distributed inference servers (multi-machine multi-GPU)\ndistServers = %s\nnumNNServerThreadsPerModel = %s\n' "$4" "$5" >> "$1"
  fi
}

mkcfg ana_eigen.cfg 1 1 ""
mkcfg ana_dist1.cfg 1 1 "127.0.0.1:$PORT1" 1
mkcfg ana_dist2.cfg 1 1 "127.0.0.1:$PORT2" 1
mkcfg game_eigen.cfg 1 200 ""
mkcfg game_dist1.cfg 1 200 "127.0.0.1:$PORT1" 1
mkcfg game_dist2.cfg 1 200 "127.0.0.1:$PORT2" 1
for f in game_eigen.cfg game_dist1.cfg game_dist2.cfg; do printf 'nnRandomize = false
' >> "$f"; done
mkcfg load_dist.cfg 4 100 "127.0.0.1:$PORT1,127.0.0.1:$PORT2" 4

ANA='boardsize 19
komi 6.5
kata-raw-nn b all
play b Q16
play w D16
kata-raw-nn w all
play b R4
play w D4
kata-raw-nn b 3
kata-raw-nn w 5
nnRandomize = false
quit
'
GAME='boardsize 19
komi 6.5
genmove b
genmove w
genmove b
genmove w
genmove b
genmove w
genmove b
genmove w
nnRandomize = false
quit
'

echo "=== [1/8] Start two distserver processes (simulating two remote GPU machines) ==="
"$SERVER_EXE" distserver -model "$MODEL" -listen 127.0.0.1:$PORT1 -gpus 0 > out/server1.log 2>&1 &
"$SERVER_EXE" distserver -model "$MODEL" -listen 127.0.0.1:$PORT2 -gpus 1 > out/server2.log 2>&1 &
sleep 3
grep -h "listening" out/server1.log out/server2.log || { echo "FAILED: servers did not start"; exit 1; }

echo "=== [2/8] Raw NN output, EIGEN direct vs distributed via SERVER 1 (must be byte-identical) ==="
printf "$ANA" | "$SERVER_EXE" gtp -model "$MODEL" -config ana_eigen.cfg 2>out/ana_eigen_err.log > out/ana_eigen.txt
printf "$ANA" | "$CLIENT_EXE" gtp -model "$MODEL" -config ana_dist1.cfg 2>out/ana_dist1_err.log > out/ana_dist1.txt
if diff out/ana_eigen.txt out/ana_dist1.txt > /dev/null; then echo "PASS (server1)"; else echo "FAIL (server1)"; diff out/ana_eigen.txt out/ana_dist1.txt | head -6; fi

echo "=== [3/8] Raw NN output, EIGEN direct vs distributed via SERVER 2 (must be byte-identical) ==="
printf "$ANA" | "$CLIENT_EXE" gtp -model "$MODEL" -config ana_dist2.cfg 2>out/ana_dist2_err.log > out/ana_dist2.txt
if diff out/ana_eigen.txt out/ana_dist2.txt > /dev/null; then echo "PASS (server2)"; else echo "FAIL (server2)"; diff out/ana_eigen.txt out/ana_dist2.txt | head -6; fi
echo "sample: $(grep -m1 -A2 "win" out/ana_dist1.txt | head -c 200)"

echo "=== [4/8] Deterministic game, direct EIGEN vs distributed via SERVER 1 ==="
printf "$GAME" | "$SERVER_EXE" gtp -model "$MODEL" -config game_eigen.cfg 2>out/game_eigen_err.log | grep -E "^(=|!)" > out/game_eigen.txt
printf "$GAME" | "$CLIENT_EXE" gtp -model "$MODEL" -config game_dist1.cfg 2>out/game_dist1_err.log | grep -E "^(=|!)" > out/game_dist1.txt
cat out/game_eigen.txt
if diff out/game_eigen.txt out/game_dist1.txt > /dev/null; then echo "PASS (server1 game identical)"; else echo "FAIL (server1 game)"; diff out/game_eigen.txt out/game_dist1.txt; fi

echo "=== [5/8] Deterministic game, direct EIGEN vs distributed via SERVER 2 ==="
printf "$GAME" | "$CLIENT_EXE" gtp -model "$MODEL" -config game_dist2.cfg 2>out/game_dist2_err.log | grep -E "^(=|!)" > out/game_dist2.txt
if diff out/game_eigen.txt out/game_dist2.txt > /dev/null; then echo "PASS (server2 game identical)"; else echo "FAIL (server2 game)"; diff out/game_eigen.txt out/game_dist2.txt; fi

echo "=== [6/8] Load spreading: 4 NN server threads across BOTH servers ==="
printf "$GAME" | "$CLIENT_EXE" gtp -model "$MODEL" -config load_dist.cfg 2>out/load_err.log | grep -E "^(=|!)" > out/game_load.txt
N1=$(grep -c "eval batch" out/server1.log); N2=$(grep -c "eval batch" out/server2.log)
echo "server1 batches: $N1   server2 batches: $N2"
if [ "$N1" -gt 0 ] && [ "$N2" -gt 0 ]; then echo "PASS: both servers served requests (multi-machine load balancing works)"; else echo "FAIL: load did not spread"; fi
echo "multi-threaded game (nondeterministic order is expected): $(grep '=' out/game_load.txt | tr '\n' ' ')"

echo "=== [7/8] Distributed benchmark: 1 vs 2 vs 4 engine threads ==="
"$CLIENT_EXE" benchmark -model "$MODEL" -config load_dist.cfg -t 1,2,4 -v 100 -n 4 -no-server-thread-test -no-half-batch-size-test 2>out/bench_dist.log | grep -E "nnEvals/s" | tail -3

echo "=== [8/8] Direct EIGEN benchmark: 1 vs 2 vs 4 threads (reference) ==="
"$SERVER_EXE" benchmark -model "$MODEL" -config game_eigen.cfg -t 1,2,4 -v 100 -n 4 -no-server-thread-test -no-half-batch-size-test 2>out/bench_eigen.log | grep -E "nnEvals/s" | tail -3

taskkill //F //IM katago.exe > /dev/null 2>&1 || true
echo "=== DONE ==="

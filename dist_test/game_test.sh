#!/usr/bin/env bash
set -u
ROOT="/c/Users/10632/Desktop/KataGo-master"
SERVER_EXE="$ROOT/build-eigen/Release/katago.exe"
CLIENT_EXE="$ROOT/build-dist/Release/katago.exe"
MODEL="/c/Users/10632/Desktop/model.bin.gz"
cd "$ROOT/dist_test"
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
genmove b
genmove w
quit
'
"$SERVER_EXE" distserver -model "$MODEL" -listen 127.0.0.1:17731 -gpus 0 > out/g_server1.log 2>&1 &
"$SERVER_EXE" distserver -model "$MODEL" -listen 127.0.0.1:17732 -gpus 1 > out/g_server2.log 2>&1 &
sleep 3
printf "$GAME" | "$SERVER_EXE" gtp -model "$MODEL" -config game_eigen.cfg 2>/dev/null | grep -E "^(=|!)" > out/g_eigen.txt
printf "$GAME" | "$CLIENT_EXE" gtp -model "$MODEL" -config game_dist1.cfg 2>/dev/null | grep -E "^(=|!)" > out/g_dist1.txt
printf "$GAME" | "$CLIENT_EXE" gtp -model "$MODEL" -config game_dist2.cfg 2>/dev/null | grep -E "^(=|!)" > out/g_dist2.txt
echo "eigen : $(tr '\n' ' ' < out/g_eigen.txt)"
echo "dist1 : $(tr '\n' ' ' < out/g_dist1.txt)"
echo "dist2 : $(tr '\n' ' ' < out/g_dist2.txt)"
if diff -q out/g_eigen.txt out/g_dist1.txt > /dev/null && diff -q out/g_eigen.txt out/g_dist2.txt > /dev/null; then
  echo "RESULT: PASS - identical 12-move games on all three engines"
else
  echo "RESULT: FAIL"; diff out/g_eigen.txt out/g_dist1.txt; diff out/g_eigen.txt out/g_dist2.txt
fi
taskkill //F //IM katago.exe > /dev/null 2>&1 || true

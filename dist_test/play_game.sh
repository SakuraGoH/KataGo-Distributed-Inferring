#!/usr/bin/env bash
# Play a fixed short deterministic game through GTP and print every genmove result.
# Usage: play_game.sh <katago-exe> <config> <label>
set -u
EXE="$1"
CFG="$2"
LABEL="$3"
MODEL="/c/Users/10632/Desktop/model.bin.gz"

printf 'boardsize 19\nkomi 6.5\nkata-genmove b\nkata-genmove w\nkata-genmove b\nkata-genmove w\nkata-genmove b\nkata-genmove w\nkata-genmove b\nkata-genmove w\nquit\n' | \
  "$EXE" gtp -model "$MODEL" -config "$CFG" 2>/dev/null | \
  grep -E "^(=|?!)? *[A-Za-z0-9]+( \[.*\])?$" | grep -v -E "KataGo|GTP|Ready|kkomi|rules" | \
  sed "s/^/[$LABEL] /"

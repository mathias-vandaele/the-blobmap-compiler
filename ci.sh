#!/usr/bin/env bash
# The milestone 2 gate (PLAN.md), run locally on every commit (SPEC §11):
#   1. unit tests, and no clippy warnings;
#   2. every scenario built and verified on the tiny tier, golden files replayed, every
#      mutant caught;
#   3. byte-identical builds with 1, 2, 8 and 24 threads;
#   4. the cells scenario built as cells and in one piece: different model_id, same
#      inputs_hash, same golden replay.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

echo "== 1. unit tests and clippy"
cargo test --workspace --quiet 2>&1 | grep -E "^test result|FAILED|panicked" || true
cargo test --workspace --quiet >/dev/null 2>&1
cargo clippy --workspace --all-targets --quiet -- -D warnings

echo "== 2. tiny tier"
cargo build --quiet --release -p blobmap-cli -p blobmap-fixtures
BLOBC="$root/target/release/blobc"
. "$root/tools/scenarios.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
target/release/fixtures "$work/in" >/dev/null
: >"$work/mutants"
fail=0
for dir in "$work"/in/*/; do
    s="$(basename "$dir")"
    for cfg in "$dir"config*; do
        v="$(basename "$cfg")"
        out="$work/m/$s-$v"
        build_scenario "$work/in" "$s" "$cfg" "$out"
        if "$BLOBC" verify "$out" --tier tiny --data "$work/in" --golden "$root/golden" --mutants >"$work/log" 2>&1; then
            echo "ok   $s $v: $(grep -c . "$work/log") lines, $(tail -1 "$work/log")"
        else
            echo "FAIL $s $v"
            cat "$work/log"
            fail=1
        fi
        grep '^mutant' "$work/log" >>"$work/mutants" || true
    done
done
# Every mutant must be caught by some scenario.
for name in $(sed -E 's/^mutant +(caught|NOT CAUGHT|not applicable) +//' "$work/mutants" | sort -u | tr ' ' '_'); do
    n="${name//_/ }"
    if ! grep -E "^mutant +caught +$n\$" "$work/mutants" >/dev/null; then
        echo "MUTANT NEVER CAUGHT: $n"
        fail=1
    fi
done
echo "mutants: $(sed -E 's/^mutant +(caught|NOT CAUGHT|not applicable) +//' "$work/mutants" | sort -u | wc -l) kinds, each caught by some scenario"

echo "== 3. determinism"
tools/determinism.sh || fail=1

echo "== 4. cells"
field() { sed -nE "s/^  \"$1\": \"([0-9a-f]+)\",?\$/\1/p" "$2/manifest.json"; }
a="$work/m/cells-config"
b="$work/m/cells-config-cells"
if [ "$(field model_id "$a")" != "$(field model_id "$b")" ] && [ "$(field inputs_hash "$a")" = "$(field inputs_hash "$b")" ]; then
    echo "cells: model_id $(field model_id "$a" | cut -c1-12) vs $(field model_id "$b" | cut -c1-12), inputs_hash $(field inputs_hash "$a" | cut -c1-12) for both; both replayed golden/tiny/cells above"
else
    echo "FAIL cells: model_id or inputs_hash"
    fail=1
fi
[ "$fail" = 0 ] && echo "GATE PASSED" || { echo "GATE FAILED"; exit 1; }

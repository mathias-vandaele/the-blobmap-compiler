#!/usr/bin/env bash
# SPEC §10, PLAN gate 3: every scenario and variant built with 1, 2, 8 and 24 threads gives
# the same SHA-256 for every file of the model directory.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cargo build --quiet --release --manifest-path "$root/Cargo.toml" -p blobmap-cli -p blobmap-fixtures
BLOBC="$root/target/release/blobc"
. "$root/tools/scenarios.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$root/target/release/fixtures" "$work/in" >/dev/null
status=0
for dir in "$work"/in/*/; do
    s="$(basename "$dir")"
    for cfg in "$dir"config*; do
        v="$(basename "$cfg")"
        ref=""
        for t in 1 2 8 24; do
            out="$work/m/$s-$v-t$t"
            RAYON_NUM_THREADS=$t OMP_NUM_THREADS=$t build_scenario "$work/in" "$s" "$cfg" "$out"
            sums="$(cd "$out" && find . -type f | sort | xargs sha256sum)"
            if [ -z "$ref" ]; then
                ref="$sums"
            elif [ "$sums" != "$ref" ]; then
                echo "NOT DETERMINISTIC: $s $v with $t threads"
                diff <(echo "$ref") <(echo "$sums") || true
                status=1
            fi
        done
        echo "deterministic: $s $v ($(echo "$ref" | wc -l) files, threads 1 2 8 24)"
    done
done
exit $status

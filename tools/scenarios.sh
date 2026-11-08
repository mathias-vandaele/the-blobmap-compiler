# Sourced by ci.sh and tools/determinism.sh: `build_scenario <inputs dir> <scenario> <config
# dir> <out>` runs `blobc build` on one fixture scenario written by the fixtures binary.
build_scenario() {
    local in="$1" s="$2" cfg="$3" out="$4" ext=()
    mkdir -p "$(dirname "$out")"
    [ -f "$in/$s/$s.poly" ] && ext=(--extent "$in/$s/$s.poly")
    "$BLOBC" build --pbf "$in/$s/$s.osm.pbf" --overture "$in/$s/overture/tiny-2026-11" \
        --config "$cfg" --out "$out" "${ext[@]}" >"$out.log" 2>&1 || { cat "$out.log"; return 1; }
}

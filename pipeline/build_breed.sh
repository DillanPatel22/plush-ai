#!/usr/bin/env bash
# Rebuild a breed's template and medium pattern from its aligned, ear-less mesh.
# Usage: build_breed.sh work/<breed> <template_dir_name> <size_label> <plush_nose_to_tail_mm> <sole_mm>
set -euo pipefail
W=$1; T=$2; SIZE=$3; MM=$4; SOLE=${5:-14}
P=$(cd "$(dirname "$0")" && pwd)
cd "$W"
python3 $P/plushify.py ears/aligned_no_ears.ply plush/plush_dense_noears.ply --sole $SOLE | tail -1
python3 $P/template_remesh.py plush/plush_dense_noears.ply $T 5000 2>&1 | grep RESULT
python3 - <<PY
import sys, json, trimesh; sys.path.insert(0, "$P"); import align_measure as am
t = trimesh.load("$T/template.obj", process=False); r, mask = am.measure(t, 1.0); am.draw(mask, r, "$T/side_silhouette.png")
r["nose_y0"] = float(t.vertices[:, 1].min()); json.dump(r, open("$T/measures.json", "w"), indent=1, default=float)
print("measures:", {k: r[k] for k in ["withers_height_mm", "length_to_height", "elbow_over_height"]})
PY
python3 $P/simplify_plush.py $T/template.obj $T/canonical.obj --body_iters 200
cp $T/canonical.obj $T/medium.obj
for i in 1 2 3; do python3 $P/fill_thin.py $T/medium.obj $T/measures.json $MM 25 $T/medium.obj > /dev/null; done
python3 $P/thin_parts.py $T/medium.obj $T/measures.json $MM 25 $T/thin_after.json
python3 $P/ear_from_trace.py ears/ear_trace_side_ref.json ref/side_72b622a5.png $T/template.obj 560 ears | tail -1
python3 $P/attach_ears_preview.py $T/canonical.obj ears $T/canonical_with_ears_preview.obj   # review renders only
K=$(python3 -c "import json; m=json.load(open('$T/measures.json')); print($MM/(m['nose_to_chest_front_mm']+m['body_length_mm']))")
python3 $P/face_points.py . $T/medium.obj $T/face_points.json $K
python3 $P/seams_standing_v1.py $T/medium.obj pieces_$T | tail -1
python3 $P/flatten_pieces.py pieces_$T $T/measures.json $MM flat_$T | tail -1
TDIR=$T python3 $P/export_pattern.py . flat_$T $SIZE $MM pattern_$T | tail -3

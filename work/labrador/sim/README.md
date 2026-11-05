# Sew-and-stuff simulation runs (pipeline/sew_stuff_sim.py)

Not calibrated yet (fabric stiffness and stuffing firmness are guesses): use for direction, not exact sizes.
The corrected pattern pieces in `corrected_stuff6/flat_comp_NOT_FOR_CUTTING/` are not for cutting until the
simulator has been checked against scanned mock-ups.

| Folder | Command (from pipeline/) |
|---|---|
| as_cut_stuff6 | `python3 sew_stuff_sim.py ../work/labrador/flat_v21_template_v04 ../work/labrador/pieces_v21_template_v04/pieces.json ../work/labrador/template_v04/measures.json OUT --iters 2500` |
| corrected_stuff6 | same with `--compensate 5 --damp 0.7` |


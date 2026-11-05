# Sew-and-stuff simulation runs (pipeline/sew_stuff_sim.py)

Not calibrated yet (fabric stiffness and stuffing firmness are guesses): use for direction, not exact sizes.
The corrected pattern pieces in `corrected_stuff6/flat_comp_NOT_FOR_CUTTING/` are not for cutting until the
simulator has been checked against scanned mock-ups.

| Folder | Command (from pipeline/) |
|---|---|
| as_cut_stuff6 | `python3 sew_stuff_sim.py ../work/english_bulldog/flat_v21 ../work/english_bulldog/pieces_v21/pieces.json ../work/english_bulldog/template/measures.json OUT --iters 2500` |
| as_cut_stuff3 | same with `--stuff 0.03` |
| corrected_stuff6 | same with `--compensate 5 --damp 0.7` |

Review figure: `work/english_bulldog/review/bulldog_sew_stuff_sim.png` (`sim_figure.py`).

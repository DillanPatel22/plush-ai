# plush-ai

Custom realistic pet plush: customer photos -> symmetrical, sewable pattern, 3D preview and workshop job card.
Full spec: "Plush AI technical spec" (Claude Docs). This file is the build agent's standing instructions.

## Rules (never break)
- Never hand-make geometry. Write and run Blender Python / Python scripts; every output is reproducible from the repo.
- Units mm. X = left/right (symmetry plane X = 0), Y = nose to tail (nose at most negative Y), Z = up (ground Z = 0).
- The dog base mesh is frozen once approved: vertex count and order never change. Breeds, poses and customer fits are reshapes.
- Edit only the +X half and mirror. Asymmetry > 0.5 mm = fail.
- Every script ends with its acceptance test and prints PASS/FAIL with numbers.
- Two FAILs in a row: stop, report the failing metric and best diagnosis.
- Commit every passing artefact with a version tag and a one-line change note.
- Only tools from the approved licence list below.

## Design rules
- Seams are never invented in production: only craftsman-approved layouts. Three poses for now: standing, sitting, lying (sphinx).
- Fur always starts white. One stock fur per plush (hair type, silicone or not, long enough for the longest zone); the craftsman trims each zone to the trim map.
- Wrinkles are needle-sculpted: mesh stays smooth; wrinkles go on a wrinkle map plus a small fabric allowance.
- Proportion target: key ratios within about +-7 %; face weighted above body.
- Fabric: knitted polyester backing, acrylic pile. Pieces laid out 0 or 180 degrees only (nap and stretch locked). Cut from the backing side.

## Defaults until the workshop confirms (print them on every pattern sheet)
- Medium plush: 350 mm nose to tail-base. Roll width 1500 mm. Seam allowance 5 mm.
- Narrowest turnable tube: 25 mm finished width. Tightest curve: 3 mm radius. Smallest piece: 25 mm across.
- Laser: DXF in mm, cut and notch on separate layers (RDWorks / LightBurn import).

## Approved tools (commercial use OK as far as published terms show; re-check before launch)
Blender / bpy (GPL software; outputs are ours), libigl (MPL-2.0), trimesh (MIT), shapely (BSD), ezdxf (MIT),
reportlab (BSD), numpy/scipy (BSD), Boundary First Flattening (MIT), PyTorch3D (BSD), SAM 3 (SAM License),
RTMPose/MMPose (Apache-2.0) + AP-10K (CC BY 4.0), Tripo H3.1 (via API/Higgsfield; confirm commercial terms).
Not allowed: SMAL/SMAL+/BARC/BITE, DeepLabCut SuperAnimal, Hunyuan3D, nvdiffrast, AnimalLift (CC BY-NC weights/data),
stock 3D marketplace models.

## Reference images
Only generated images or openly licensed photos (record licence + author in `work/<breed>/ref/**/licences.json`).
CC BY-SA photos are used for measurement only, never as generation input or in deliverables.

## Layout
- library/dog/        frozen base mesh, rig, poses, breed shapes (versioned)
- layouts/            frozen seam layouts per species and pose
- pipeline/           scripts: shape, plushify, seams, flatten, export, checks
- factors/            correction factors (versioned)
- work/<breed>/       per-breed working files, references, renders, review sheets
- tests/              acceptance tests

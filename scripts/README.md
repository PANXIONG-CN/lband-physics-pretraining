# Entry points

The compact-release workflows are `verify_release.py`,
`recalculate_frozen_results.py`, `evaluate_physics_offset_baseline.py`,
`render_revision_figures.py`, and `verify_stage_continuity.py`.
They use only the shipped compact paths. See the root reproducibility guide.

Other files retain original experiment and extended-control implementations.
Some require official raw products or extra historical outputs, as explicitly
listed in that guide. Their presence does not imply those missing controls
were replayed. Original optimizer behavior is intentionally preserved.

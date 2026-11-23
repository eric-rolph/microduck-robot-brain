"""Console script entrypoints for microduck-brain."""

from __future__ import annotations


def run_physical() -> None:
    from scripts.run_physical_microduck import main
    main()


def run_sim() -> None:
    from scripts.run_mujoco_sim import main
    main()


def run_eval() -> None:
    from scripts.run_brain_eval_suite import main
    main()


def check_duck() -> None:
    from scripts.check_physical_duck import main
    main()

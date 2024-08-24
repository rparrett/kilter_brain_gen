from pathlib import Path

TRAINING_DIR = "models"


def get_next_subdir_name():
    existing_dirs = [d for d in Path(TRAINING_DIR).iterdir() if d.is_dir()]
    existing_nums = [
        int(d.name.split("_")[-1])
        for d in existing_dirs
        if all(c.isdigit() for c in d.name.split("_")[-1])
    ]
    return (
        f"climb_clm_run_{max(existing_nums) + 1:03d}"
        if existing_nums
        else "climb_clm_run_000"
    )


def get_latest_checkpoint_path():
    existing_dirs = [
        d
        for d in Path(TRAINING_DIR).rglob("climb_clm_run_*")
        if d.is_dir() and all(c.isdigit() for c in d.name.split("_")[-1])
    ]
    latest_dir = max(existing_dirs, key=lambda d: int(d.name.split("_")[-1]))
    print("latest_dir:", latest_dir)
    latest_checkpoint_in_dir = max(
        latest_dir.rglob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return latest_checkpoint_in_dir

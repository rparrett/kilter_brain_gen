from pathlib import Path


def find_latest_checkpoint(base_dir):
    """Find the most recent checkpoint directory in the given base directory."""
    base_path = Path(base_dir)

    # Find all checkpoint directories recursively
    checkpoint_dirs = []
    for checkpoint_dir in base_path.rglob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            checkpoint_dirs.append(checkpoint_dir)

    if not checkpoint_dirs:
        return None

    # Sort by modification time and return the most recent
    latest = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    return latest

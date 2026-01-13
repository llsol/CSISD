import yaml
from pathlib import Path


def get_tonic_from_recording_id(
        recording_id: str,
        tonics_yaml: str | Path = "data/corpus/tonics.yaml"
) -> float:
    
    tonics_yaml = Path(tonics_yaml)

    if not tonics_yaml.exists():
        raise FileNotFoundError(f"Missing tonics file: {tonics_yaml}")

    with tonics_yaml.open("r") as f:
        tonics = yaml.safe_load(f)

    if recording_id not in tonics:
        raise KeyError(
            f"Recording id '{recording_id}' not found in {tonics_yaml}. "
            f"Available keys: {list(tonics.keys())}"
        )

    return float(tonics[recording_id])

from pathlib import Path

target_dir = Path("data/interim/scms_pitch")

for f in target_dir.glob("*swiftf0-scratch_raw.npy"):
    print(f"Deleting: {f}")
    f.unlink()

print("Done.")
from pathlib import Path
import polars as pl

from src.io.pitch_io import save_preprocessed_pitch
from src.io.pitch_io import load_pitch_file   
from src.preprocessing.pitch_preprocessing import preprocess_pitch
from settings import SARASUDA_VARNAM





def preprocess_one_recording(recording_id):
    
    raw_path = Path(f"data/corpus/{recording_id}/raw/{recording_id}_pitch_ftanet.tsv")
    df_raw = load_pitch_file(raw_path, column_names=["time_rel_sec", "f0_Hz"])
    df_prep = preprocess_pitch(df_raw)

    save_preprocessed_pitch(df_prep, recording_id, debug=False, create_tsv=True)





def preprocess_all_recordings():
    for rec in SARASUDA_VARNAM:
        print(f" Preprocessant {rec}...")
        preprocess_one_recording(rec)
    print("\n All recordings preprocessed.")





if __name__ == "__main__":

    preprocess_all_recordings()

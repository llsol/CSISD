
# fes un zip entre dues llistes random
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
zipped = zip(list1, list2)
print(list(zipped))






'''
import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec


def load_annotation_tsv(
        file_path: Path | str | None,
        recording_id: str | None = None,
        engine='polars',
        sep='\t',
        annotation_type: str = 'svara',
        column_names=None
):
    """

    """
    if file_path is None and recording_id is None:
        raise ValueError("Either file_path or recording_id must be provided.")

    if file_path is not None:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        ext = file_path.suffix.lower()


        if engine == 'polars':

            if ext == '.parquet':
                df = pl.read_parquet(file_path)

            elif ext == '.tsv':
                if column_names is None:
                    df = pl.read_csv(file_path, separator=sep, has_header=True)

                else:
                    df = pl.read_csv(file_path, separator=sep, has_header=False)
                    df = df.rename({old: new for old, new in zip(df.columns, column_names)})


            else:
                raise ValueError(f"Unsupported file extension: {ext}")


        elif engine == 'pandas':

            if ext == '.parquet':
                df = pd.read_parquet(file_path)

            elif ext == '.tsv':
                if column_names is None:
                    df = pd.read_csv(file_path, sep=sep, header=0)

                else:
                    df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            raise ValueError("Engine must be 'polars' or 'pandas'.")
    
    else:
        
        if isinstance(recording_id, str):

            dir_path = Path('data' / 'corpus' / recording_id / 'annotations')
            path_end = f'ann_{annotation_type}.tsv'

            for file in dir_path.iterdir().rglob('*_ann_*'):
                if str(file).endswith(path_end):
                
                    file_path = Path(file)
                    if engine == 'polars':

                        if ext == '.parquet':
                            df = pl.read_parquet(file_path)

                        elif ext == '.tsv':
                            if column_names is None:
                                df = pl.read_csv(file_path, separator=sep, has_header=True)

                            else:
                                df = pl.read_csv(file_path, separator=sep, has_header=False)
                                df = df.rename({old: new for old, new in zip(df.columns, column_names)})


                        else:
                            raise ValueError(f"Unsupported file extension: {ext}")


                    elif engine == 'pandas':
                    
                        if ext == '.parquet':
                            df = pd.read_parquet(file_path)

                        elif ext == '.tsv':
                            if column_names is None:
                                df = pd.read_csv(file_path, sep=sep, header=0)

                            else:
                                df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)

                        else:
                            raise ValueError(f"Unsupported file extension: {ext}")

                    else:
                        raise ValueError("Engine must be 'polars' or 'pandas'.")
                    

    return df



'''








'''import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec


def load_annotation_tsv(
        file_path: Path | str | None,
        recording_id: str | None = None,
        engine='polars',
        sep='\t',
        annotation_type: str = 'svara_marks',
):
    """

    """
    if file_path is None and recording_id is None:
        raise ValueError("Either file_path or recording_id must be provided.")

    if file_path is not None:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        ext = file_path.suffix.lower()


        if engine == 'polars':

            if ext == '.parquet':
                df = pl.read_parquet(file_path)
                df['Begin time'] = time_str_to_sec(df['Begin Time'])
                df['End time'] = time_str_to_sec(df['End Time'])

            elif ext == '.tsv':
                if column_names is None:
                    df = pl.read_csv(file_path, separator=sep, has_header=True)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)
                else:
                    df = pl.read_csv(file_path, separator=sep, has_header=False)
                    df = df.rename({old: new for old, new in zip(df.columns, column_names)})
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")


        elif engine == 'pandas':

            if ext == '.parquet':
                df = pd.read_parquet(file_path)
                df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                df['End time'] = df['End Time'].apply(time_str_to_sec)

            elif ext == '.tsv':
                if column_names is None:
                    df = pd.read_csv(file_path, sep=sep, header=0)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)
                else:
                    df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            raise ValueError("Engine must be 'polars' or 'pandas'.")
    
    else:
        
        if isinstance(recording_id, str):

            dir_path = Path('data' / 'corpus' / recording_id / 'annotations')

            for file in dir_path.iterdir():
                for  tag in str(file).split('_'):
                    if tag == 'ann':
                        
                        break


    return df'''
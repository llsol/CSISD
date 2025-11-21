

def time_str_to_sec(time_str: str) -> float:
    
    '''Convert a time string in the format "MM:SS.sss" or "HH:MM:SS.sss" to seconds.'''
    
    parts = time_str.split(':')
    parts = [float(part.strip()) for part in parts]
    
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    
    else:
        raise ValueError("Invalid time format. Expected 'MM:SS.sss' or 'HH:MM:SS.sss'.")
    



def validate_columns(df, required_columns):

    '''Check if the DataFrame contains the required columns.'''

    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
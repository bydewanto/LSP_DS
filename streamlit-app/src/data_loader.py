import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    return df

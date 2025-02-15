import pandas as pd
def load_data(filename):
    df=pd.DataFrame
    raw_data=pd.read_parquet(filename)
    df=raw_data[raw_data['domain'].isin(["blocksworld", "blocksworld_3"])]
    return df
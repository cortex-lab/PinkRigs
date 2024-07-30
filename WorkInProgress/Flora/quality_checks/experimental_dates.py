# %% 
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"C:\Users\Flora\Documents\Github\PinkRigs") 


savepath = [r'D:\VideoAnalysis\active_dataset.csv',
            r'D:\VideoAnalysis\naive_dataset.csv',
]

rr  = []
for s in savepath:
    df = pd.read_csv(s)

    df['expDate'] = pd.to_datetime(df['expDate'])
    # Grouping by 'subject' and aggregating first and last expDate
    result = df.groupby('subject')['expDate'].agg(['first', 'last']).reset_index()

    rr.append(result)

all_res = pd.concat(rr)
all_res.to_csv('D:\experimental_dates_ephys.csv')
# %%

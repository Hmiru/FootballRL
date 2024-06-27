import pandas as pd
import os
data=pd.read_csv("total_state_data_with_paths.csv", index_col=0)
print(data['outcome_name'].value_counts())

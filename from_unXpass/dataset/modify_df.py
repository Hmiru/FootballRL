import pandas as pd
data=pd.read_csv("total_data_with_state_label_mask.csv", index_col=0)
data= data.fillna(0)
data.to_csv("total_data_with_state_label_mask.csv")


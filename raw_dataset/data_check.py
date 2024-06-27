import pandas as pd
data= pd.read_csv("total_state_data.csv", index_col=0)
print(data.sort_values('end_y', ascending=True)[['end_x', 'end_y']].head(10))

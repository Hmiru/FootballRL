import pandas as pd

if __name__=="__main__":
     data1=pd.read_csv("WC_EU_LEV_data.csv", index_col=0)
     data2=pd.read_csv("Leverkusen_total_data_with_state_label_mask.csv", index_col=0)

     data=pd.concat([data1, data2], ignore_index=True)
     data.to_csv("WC_EU_LEV_data.csv")
     data=pd.read_csv("WC_EU_LEV_data.csv", index_col=0)
    # # print(data)
     print(data['outcome_name'].value_counts())
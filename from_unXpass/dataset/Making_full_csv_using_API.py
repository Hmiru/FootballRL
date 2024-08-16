from utils import *
from flatten import flatten_columns

parser = Sbopen()
def final_df(competition_id, season_id):

    df_match = parser.match(competition_id=competition_id, season_id=season_id)
    all_matches_df = []
    for id in (df_match['match_id']):
        print(id)
        df_matches={}
        df_matches[id] = {}
        df_matches[id]['event'] = parser.event(id)

        df_frame, df_visible = parser.frame(id)
        print(df_frame.shape)
        specific_id_row= df_frame.loc[df_frame['id'] == 'b4fc9de7-9029-4f18-bd6c-3d2a3c5e276a']
        print(specific_id_row.to_string())


    #     df_evaluate = df_matches[id]['event'][0]  # .copy()를 사용하여 명확하게 복사본을 만들어 작업
    #
    #     evaluate_mask = (df_evaluate['type_name'].isin(['Pass'])) & (df_evaluate['period'] <= 4)
    #     df_evaluate = df_evaluate[evaluate_mask]
    #     df_summary=df_evaluate[['id','index','type_name','x','y','end_x','end_y', 'outcome_name']]
    #     print(df_summary.shape)
    #
    #     result_list = []
    #     for x in range(len(df_summary)):
    #         row = df_summary.iloc[x]
    #         mid_df = flatten_columns(row, df_frame)
    #         result_list.append(mid_df)
    #
    #     match_df = pd.concat(result_list, axis=1).transpose()
    #     all_matches_df.append(match_df)
    # #
    # # 리스트를 DataFrame으로 병합
    # final_df = pd.concat(all_matches_df, ignore_index=True)
    # print(final_df)
    # print(final_df.columns)
    # print(final_df.shape)
    # return final_df
if __name__=="__main__":
    final_df(competition_id=43, season_id=106)  # World Cup 2022
    # final_df = final_df(competition_id=43, season_id=106) # World Cup 2022
    # final_df.to_csv("total_data_with_state_label_mask.csv")



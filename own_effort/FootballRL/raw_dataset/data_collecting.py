import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Sbopen,Pitch, VerticalPitch, FontManager

parser = Sbopen()
## World Cup 2022
df_match = parser.match(competition_id=43, season_id=106)

## Euro 2020
## df_match = parser.match(competition_id=55, season_id=43)
## World Cup 2022
# df_matches = {}
for i, id in enumerate(df_match['match_id']):
     df_matches[id] = {}
     df_matches[id]['event'], df_matches[id]['related'], df_matches[id]['freeze'], df_matches[id]['tactic'] = parser.event(id)
df_frame, df_visible = parser.frame(3869685)
#first game of WC 2022

# exploring the data
print(df_matches[3869685]['event'].iloc[137:178,:])
from mplsoccer import Pitch
import numpy as np
def draw_grid_on_pitch(actor_coor, team_coor, oppo_coor):
# setup the full pitch
  pitch = Pitch(pitch_type='statsbomb', pad_bottom=1, label=True, axis=True,
                half=False,  # full pitch
                pitch_length=105, pitch_width=68,  # using 105x68 dimensions
                goal_type='box',
                goal_alpha=0.8, pitch_color='#22312b', line_color='#c7d5cc')

  fig, ax = pitch.draw(figsize=(12, 7))

  # 축구장을 수평으로 나누기
  h_lines = np.linspace(0, 80, 21)  # 축구장의 y-좌표를 17개 구역으로 나눔
  for y in h_lines:
      ax.axhline(y, color='white', linestyle='-',linewidth=0.5)  # 수평선 추가

  # 축구장을 수직으로 나누기
  v_lines = np.linspace(0, 120, 31)  # 축구장의 x-좌표를 27개 구역으로 나눔
  for x in v_lines:
      ax.axvline(x, color='white', linestyle='-', linewidth=0.5)  # 수직선 추가


  actor = pitch.scatter(*zip(*actor_coor), c='#ba4f45', marker='o', ax=ax, s=70, edgecolors='white', linewidths=0.5)
  teammates = pitch.scatter(*zip(*team_coor), c='#f4cccc', marker='o', ax=ax, s=70, edgecolors='white', linewidths=0.5)
  oppopnents = pitch.scatter(*zip(*oppo_coor), c='#3d85c6', marker='o', ax=ax, s=70, edgecolors='white', linewidths=0.5)

  all_players = actor_coor + team_coor
  for coor in all_players:
    #print(coor)
    row = {
          'under_pressure': False,
          'shot_first_time': True,
          'x': coor[0],
          'y': coor[1],
          'body_part_name': 'Left Foot',
          'technique_name': 'Normal',
          'sub_type_name': 'Open Play',
          'play_pattern_name': 'From Counter'
  }
    xg = calculate_xg_adv(row)
    ax.text(coor[0], coor[1] + 0, f"{xg:.2f}", color='white', ha='center', va='bottom', fontsize=12, zorder=2)

  ax.tick_params(colors='black', which='both')  # 모든 눈금의 색상을 노란색으로 변경


  # 레전드 핸들러 생성
  from matplotlib.lines import Line2D
  legend_elements = [Line2D([0], [0], marker='o', color='w', label='actor', markerfacecolor='#ba4f45', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='teammates', markerfacecolor='#f4cccc', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='opponents', markerfacecolor='#3d85c6', markersize=10)]


  # 레전드 추가
  ax.legend(handles=legend_elements, loc='upper left')  # 상단 왼쪽으로 레전드 위치 조정
  plt.show()


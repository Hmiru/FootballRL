
from utils import *
class RowToState:
    def __init__(self, row):
        self.goal_center = np.array([120, 40])
        self.goal_left = np.array([120, 44])
        self.goal_right = np.array([120, 36])
        self.row = row
        self.actor_x = row['actor_x']
        self.actor_y = row['actor_y']
        self.feature_stack = []
        self.outcome_name = ['nan', 'Incomplete', 'Out', 'Unknown', 'Pass Offside', 'success']
        self.enc = OneHotEncoder(sparse_output=False)
        self.enc.fit(np.array(self.outcome_name).reshape(-1, 1))

    def visualize(self):
        num_features = len(self.feature_stack)
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols

        plt.figure(figsize=(15, 5 * num_rows))
        for i, feature in enumerate(self.feature_stack):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(feature, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Feature {i + 1}')
        plt.tight_layout()
        plt.show()

    def calculate_distance(self, x, y, target_x, target_y):
        return np.sqrt((x - target_x)**2 + (y - target_y)**2)

    def distance(self, desde):
        create_array = np.zeros((80, 120), dtype=float)
        for y in range(create_array.shape[0]):
            for x in range(create_array.shape[1]):
                if desde == "ball":
                    dist = self.calculate_distance(x, y, self.actor_x, self.actor_y)
                elif desde == "goal":
                    dist = self.calculate_distance(x, y, self.goal_center[0], self.goal_center[1])
                create_array[y, x] = dist
        self.feature_stack.append(create_array)
        return self

    def angle(self, focus, que):
        create_array = np.zeros((80, 120), dtype=float)

        # 각도 계산 및 저장
        for y in range(create_array.shape[0]):
            for x in range(create_array.shape[1]):
                my_position = np.array([x, y])

                if focus == "goalpost":
                    # 벡터 계산
                    upper_vector = self.goal_left - my_position
                    lower_vector = self.goal_right - my_position
                elif focus == "ball":
                    ball_position = np.array([self.row['actor_x'], self.row['actor_y']])
                    # 벡터 계산
                    upper_vector = ball_position - my_position
                    lower_vector = self.goal_center - my_position

                # 벡터 크기 계산
                upper_norm = np.linalg.norm(upper_vector)
                lower_norm = np.linalg.norm(lower_vector)

                # 코사인 값 계산
                if upper_norm != 0 and lower_norm != 0:
                    if que == "cosine":
                        tri_func = np.dot(upper_vector, lower_vector) / (upper_norm * lower_norm)
                    elif que == "angle":
                        cos_angle = np.dot(upper_vector, lower_vector) / (upper_norm * lower_norm)
                        tri_func = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    elif que == "sine":
                        cross_product = upper_vector[0] * lower_vector[1] - upper_vector[1] * lower_vector[0]
                        tri_func = cross_product / (upper_norm * lower_norm)
                else:
                    tri_func = 0

                create_array[y, x] = tri_func

        self.feature_stack.append(create_array)
        return self

    def player_coor(self, row, role):
        create_array = np.zeros((80, 120), dtype=int)
        positions = []

        if role == "actor":
            positions.append((row['actor_x'], row['actor_y']))

        elif role == "teammate":
            # actor와 teammate들의 위치를 1로 설정
            positions.extend(
                (row[f'teammate_{i}_x'], row[f'teammate_{i}_y'])
                for i in range(1, 11)
                if not pd.isnull(row[f'teammate_{i}_x']) and not pd.isnull(row[f'teammate_{i}_y'])
            )
        else:
            # keeper와 opponent들의 위치를 1로 설정
            if not pd.isnull(row['keeper_x']) and not pd.isnull(row['keeper_y']):
                positions.append((row['keeper_x'], row['keeper_y']))
            positions.extend(
                (row[f'opponent_{i}_x'], row[f'opponent_{i}_y'])
                for i in range(1, 11)
                if not pd.isnull(row[f'opponent_{i}_x']) and not pd.isnull(row[f'opponent_{i}_y'])
            )

        for x, y in positions:
            x, y = int(x), int(y)
            if 0 <= y < 80 and 0 <= x < 120:
                create_array[y, x] = 1
            else:
                print(f"Index out of bounds for x={x}, y={y}")
        self.feature_stack.append(create_array)
        return self

    def outcome_label(self):
        outcome_name = str(self.row["outcome_name"])
        out_label=np.array(0 if outcome_name == "success" else 1)
        return out_label

    def destination_label(self):
        end_x = self.row["end_x"]
        end_y = self.row["end_y"]

        dest_label = np.array([end_x, end_y]).reshape(1, -1)
        return dest_label


    def stack_features(self):
        return np.stack(self.feature_stack, axis=-1)


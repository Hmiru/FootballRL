from utils import *
from network import SuccessPredictionNetwork

def visualize_success_surface(success_surface):
    """
    시각화 함수: 주어진 성공 확률 맵을 시각화합니다.

    Parameters:
    success_surface (numpy.ndarray): 성공 확률 맵, 크기는 (80, 120)

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(success_surface, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Success Probability')
    plt.title('Success Surface')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()

def generate_success_surface(policy_network, state):
    """
    주어진 상태에서 성공 확률 맵을 생성합니다.

    Parameters:
    policy_network (torch.nn.Module): 정책 네트워크
    state (numpy.ndarray): 현재 상태, 크기는 (80, 120, 11)

    Returns:
    numpy.ndarray: 성공 확률 맵, 크기는 (80, 120)
    """
    # 상태를 텐서로 변환하고 차원을 재구성합니다.
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

    # 네트워크를 통해 성공 확률 예측
    with torch.no_grad():
        success_probabilities = policy_network(state_tensor).squeeze().cpu().numpy()

    return success_probabilities

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (80, 120, 11)
    output_shape = (80, 120)

    net = SuccessPredictionNetwork(input_shape, output_shape).to(device)
    net.load_state_dict(torch.load('success_prediction_network.pth', map_location=device))  # 학습된 모델의 경로 지정
    net.eval()

    data = pd.read_csv("../dataset/total_state_data_with_paths.csv", index_col=0)

    # 두 번째 데이터 샘플을 가져오기
    second_sample = data.iloc[400]
    sample_state = np.load(second_sample['state_path'])

    # 성공 확률 맵 생성
    output = generate_success_surface(net, sample_state)

    print(output)
    print(output.shape)

    visualize_success_surface(output)


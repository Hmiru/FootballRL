from trainer import OfflineLearner
from dataset import SoccerDataset
from env import SoccerEnv
from network import SuccessPredictionNetwork
from utils import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_shape = (80, 120, 11)
    output_shape = (80, 120)
    net = SuccessPredictionNetwork(input_shape, output_shape).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    learner = OfflineLearner(net, optimizer, device=device, pos_weight=10.3)
    data = pd.read_csv("../dataset/total_state_data_with_paths.csv", index_col=0)

    dataset = SoccerDataset(data)
    env = SoccerEnv(dataset)
    learner.train(env, num_episodes=1)

    torch.save(net.state_dict(), "success_prediction_network.pth")
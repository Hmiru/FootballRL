from utils import *
class OfflineLearner:
    def __init__(self, policy_network, optimizer, pos_weight=4.3,gamma=0.9, batch_size=64,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.device=device
        self.pos_weight= torch.tensor(pos_weight,device=device)

    def update_policy(self, states, actions, rewards):
        self.policy_network.train()
        dataset = list(zip(states, actions, rewards))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for batch_states, batch_actions, batch_rewards in dataloader:
            batch_states = np.array(batch_states, dtype=np.float32)
            batch_states = torch.FloatTensor(batch_states).permute(0, 3, 1, 2).to(self.device)
            batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)

            batch_rewards=(batch_rewards+1)/2
            probs = self.policy_network(batch_states)
            probs = probs.view(-1, 80 * 120)
            selected_probs = probs[range(probs.size(0)), batch_actions]

            loss = F.binary_cross_entropy(selected_probs, batch_rewards,weight=self.pos_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, num_episodes=1000):
        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            state = env.reset()
            states = []
            actions = []
            rewards = []
            done = False

            with trange(len(env.dataset), desc=f"Episode {episode + 1}", leave=False) as t:
                while not done:
                    states.append(state)

                    action = env.get_action()
                    print(action)
                    if action is None:
                        print(f"Invalid action at episode {episode}, state index {len(states)}")
                        break
                    actions.append(action)
                    next_state, reward, done, _ = env.step()
                    rewards.append(reward)
                    state = next_state
                    t.update(1)
                    if done:
                        break

            if not states:
                print(f"No valid states at episode {episode}")
                continue

            self.update_policy(states, actions, rewards)

            with torch.no_grad():
                test_state = torch.FloatTensor(states[0]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                test_output = self.policy_network(test_state)
                print("Sample output:", test_output.view(-1, 80, 120)[0].cpu().numpy())

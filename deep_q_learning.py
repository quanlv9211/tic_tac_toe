import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, n_action, capacity, batch_size, device="auto"):
        self.capacity = capacity
        self.batch_size = batch_size
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            device = self.device
        else:
            self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32

        self.obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.masks = np.empty((capacity, n_action), dtype=bool)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, dones, mask, info=None):
        """Add a new transition to replay buffer"""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], dones)
        np.copyto(self.masks[self.idx], mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        """Sample batch of Transitions with batch_size elements.
        Return a named tuple with 'states', 'actions', 'rewards', 'next_states' and 'dones'.
        """
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return (obses, actions, rewards, next_obses, dones, masks)


def str2vec(state):
    """convert string representation of the board to a vector"""
    state = list(state)

    symbol2int = {"X": 1, "O": -1}  # default be 0

    state = [symbol2int.get(s, 0) for s in state]

    return np.array(state, dtype=np.float32)


class DeepQlearningRunner(object):
    def __init__(self, args):
        self.args = args
        # self.player1 = load_model(self.args).to(self.args.device)
        # self.player2 = load_opponent_model(self.args).to(self.args.device)
        self.player1 = load_model(self.args)
        self.player2 = load_opponent_model(self.args)
        dict_args = vars(args)
        self.simulator = Simulator(
            **dict_args, player1=self.player1, player2=self.player2
        )

        print("Player 2 is replaced with deep Q learning agent")
        self.gamma = 0.99
        self.batch_size = 128
        self.device = "cpu"

    def train_dql(self):
        input_dim = (
            self.simulator.current_board.width * self.simulator.current_board.height
        )

        # at each steps, the number of possible actions equal the number of empty square,
        # which is at most the size of the board at the beginning of the game
        n_actions = (
            self.simulator.current_board.width * self.simulator.current_board.height
        )

        self.n_actions = n_actions
        tau = 0.98

        # the q function using a neural network
        self.q_values = QNetwork(input_dim, n_actions).to(self.device)
        target_q_values = QNetwork(input_dim, n_actions).to(self.device)
        target_q_values.load_state_dict(self.q_values.state_dict())

        import torch.optim as optim

        optimizer = optim.Adam(self.q_values.parameters(), lr=1e-2)

        self.buffer = ReplayBuffer(
            input_dim, n_actions, 100000, self.batch_size, self.device
        )
        ret = []

        for eps in range(5000):
            state, avail_action = self.simulator.reset()
            state = str2vec(state)
            rewards_all = []
            while not self.simulator.current_board.gameover():

                if np.random.binomial(1, 0.01) == 1:
                    action = np.random.choice(avail_action)
                else:
                    state = torch.as_tensor(state, device=self.device)
                    with torch.no_grad():
                        q = self.q_values(state).cpu().numpy()

                    q[
                        avail_action
                    ] += 100  # we choose greedy action from available actions only
                    action = np.argmax(q)
                next_state, reward, done, avail_action = self.simulator.step(action)
                next_state = str2vec(next_state)

                mask = np.zeros(input_dim, dtype=bool)
                mask[avail_action] = 1

                self.buffer.add(state, action, reward, next_state, done, mask)
                del mask

                rewards_all.append(reward)
                state = next_state

                if (
                    eps < 50
                ):  # do not train when there is not enough data in the replay buffer
                    continue

                # Training
                obses, actions, rewards, next_obses, dones, masks = self.buffer.sample()

                with torch.no_grad():
                    next_q = target_q_values(next_obses)
                    assert next_q.shape == masks.shape
                    next_q[~masks] -= 100
                    next_q_max = next_q.max(dim=-1)[0]
                    next_q_max = torch.clamp(next_q_max, min=-1)
                    assert (
                        next_q_max.shape == dones.flatten().shape
                    ), f"{next_q_max.shape} {dones.flatten().shape}"
                    td_target = rewards.flatten() + self.gamma * next_q_max * (
                        1 - dones.flatten()
                    )
                old_val = self.q_values(obses).gather(1, actions.long()).squeeze()
                loss = F.mse_loss(td_target, old_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if eps % 10 == 0:

                    for target_network_param, q_network_param in zip(
                        self.q_values.parameters(), target_q_values.parameters()
                    ):
                        target_network_param.data.copy_(
                            tau * q_network_param.data
                            + (1.0 - tau) * target_network_param.data
                        )
            if eps % 200 == 0:
                print("Avg Rewards:", np.mean(ret[-20:]))

            ret.append(np.sum(rewards_all))

        return ret

    def test(self):
        logger.info("Start testing")
        # self.player1.load_policy()
        # self.player2.load_policy()
        player1_win = 0.0
        player2_win = 0.0
        turns = self.args.test_games
        for _ in range(turns):
            state, avail_action = self.simulator.reset()
            state = str2vec(state)

            while not self.simulator.current_board.gameover():

                if np.random.binomial(1, 0.001) == 1:
                    action = np.random.choice(avail_action)
                else:
                    state = torch.as_tensor(state, device=self.device)
                    with torch.no_grad():
                        q = self.q_values(state).cpu().numpy()

                    q[
                        avail_action
                    ] += 1000  # we choose greedy action from available actions only
                    action = np.argmax(q)
                state, reward, done, avail_action = self.simulator.step(action)
                state = str2vec(state)

            if self.simulator.current_board.iswin("X"):
                player1_win += 1
            if self.simulator.current_board.iswin("O"):
                player2_win += 1

        logger.info(
            "%d games, player 1 winrate %.02f, player 2 (Q Learning agent) winrate %.02f"
            % (turns, player1_win / turns, player2_win / turns)
        )


if __name__ == "__main__":
    from config import args
    from util import set_random, init_logger, prepare_dir, logger
    from load_agent import load_model, load_opponent_model
    from tic_tac_toe.simulator.simulator import Simulator
    import warnings

    warnings.filterwarnings("ignore")
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + "result.txt")
    runner = DeepQlearningRunner(args)
    ret = runner.train_dql()
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ret = savgol_filter(ret, 300, 1)
    plt.plot(ret)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("deep Q learning")
    plt.savefig("dql.png")
    runner.test()

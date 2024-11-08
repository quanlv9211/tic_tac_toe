import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim),
        )

    def forward(self, x, avail_action=None):
        logits = self.network(x)
        if avail_action is not None:
            logits[~avail_action.bool()] -= 1e8
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist


def str2vec(state):
    """convert string representation of the board to a vector"""
    state = list(state)

    symbol2int = {"X": 1, "O": -1}  # default be 0

    state = [symbol2int.get(s, 0) for s in state]

    return np.array(state, dtype=np.float32)


class ReinforceRunner(object):
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

        print("Player 2 is replaced with MonteCarlo agent")
        self.gamma = 0.99
        self.device = "cpu"

    def train_reinforce(self):
        input_dim = (
            self.simulator.current_board.width * self.simulator.current_board.height
        )

        # at each steps, the number of possible actions equal the number of empty square,
        # which is at most the size of the board at the beginning of the game
        n_actions = (
            self.simulator.current_board.width * self.simulator.current_board.height
        )

        self.n_actions = n_actions

        self.policy = Policy(input_dim, n_actions).to(self.device)
        import torch.optim as optim

        optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        ret = []

        for eps in range(10000):
            state, avail_action = self.simulator.reset()
            state = str2vec(state)
            states = [copy.deepcopy(state)]
            actions = []
            rewards = []
            while not self.simulator.current_board.gameover():

                with torch.no_grad():
                    state = torch.as_tensor(state, device=self.device)
                    action, dist = self.policy(state)
                    action = action.cpu().numpy()
                state, reward, done, avail_action = self.simulator.step(action)
                state = str2vec(state)

                states.append(copy.deepcopy(state))
                rewards.append(reward)
                actions.append(action)

            reward_to_go = np.zeros(len(rewards) + 1)
            # the last reward to go is 0, this is similar to (1-done) in q learning
            for t in reversed(range(len(rewards))):
                reward_to_go[t] = rewards[t] + self.gamma * reward_to_go[t - 1]

            # optimize
            states = torch.as_tensor(np.array(states[:-1]), device=self.device)
            reward_to_go = torch.as_tensor(reward_to_go[:-1], device=self.device)
            actions = torch.as_tensor(np.array(actions), device=self.device).long()
            _, dist = self.policy(states)

            log_prob = dist.log_prob(actions)
            # print(reward_to_go)
            assert log_prob.shape == reward_to_go.shape

            loss = -torch.mean(log_prob * reward_to_go)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ret.append(np.sum(rewards))

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

                with torch.no_grad():
                    state = torch.as_tensor(state, device=self.device)
                    action, dist = self.policy(state)
                    action = action.cpu().numpy()
                state, reward, done, avail_action = self.simulator.step(action)
                state = str2vec(state)

            if self.simulator.current_board.iswin("X"):
                player1_win += 1
            if self.simulator.current_board.iswin("O"):
                player2_win += 1

        logger.info(
            "%d games, player 1 winrate %.02f, player 2 (MC agent) winrate %.02f"
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
    runner = ReinforceRunner(args)
    ret = runner.train_reinforce()
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ret = savgol_filter(ret, 300, 1)
    plt.plot(ret)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("REINFORCE")
    plt.savefig("reinforce.png")
    runner.test()

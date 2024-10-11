import os
import sys
import numpy as np
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class QlearningRunner(object):
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

        print("Player 2 is replaced with Q learning agent")
        self.gamma = 0.99

    def train_mc(self):
        # each square in a board can be in 3 sates: empty, X, and O
        n_states = 3 ** (
            self.simulator.current_board.width * self.simulator.current_board.height
        )
        # at each steps, the number of possible actions equal the number of empty square,
        # which is at most the size of the board at the beginning of the game
        n_actions = (
            self.simulator.current_board.width * self.simulator.current_board.height
        )

        self.n_actions = n_actions

        # the q value stored in a table
        self.q_values = {}
        ret = []

        for eps in range(20000):
            state, avail_action = self.simulator.reset()
            rewards = []
            while not self.simulator.current_board.gameover():

                if np.random.binomial(1, 0.05) == 1:
                    action = np.random.choice(avail_action)
                else:
                    q = copy.deepcopy(
                        self.q_values.get(state, np.random.randn(n_actions))
                    )
                    # self.q_values[state] = q

                    q[
                        avail_action
                    ] += 100  # we choose greedy action from available actions only
                    action = np.argmax(q)
                next_state, reward, done, avail_action = self.simulator.step(action)

                q = copy.deepcopy(self.q_values.get(state, np.zeros(n_actions)))
                next_q = np.max(q) if not done else 0
                q[action] = q[action] + 0.1 * (reward + self.gamma * next_q - q[action])
                self.q_values[state] = q
                state = next_state
                rewards.append(reward)

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

            while not self.simulator.current_board.gameover():

                if np.random.binomial(1, 0.001) == 1:
                    action = np.random.choice(avail_action)
                else:
                    q = self.q_values.get(state, None)
                    if q is None:
                        print("not found")
                        q = np.random.randn(self.n_actions)
                    self.q_values[state] = q
                    q = copy.deepcopy(q)

                    q[
                        avail_action
                    ] += 100  # we choose greedy action from available actions only
                    action = np.argmax(q)
                state, reward, done, avail_action = self.simulator.step(action)

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
    runner = QlearningRunner(args)
    ret = runner.train_mc()
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ret = savgol_filter(ret, 300, 1)
    plt.plot(ret)
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("Q learning")
    plt.savefig("ql.png")
    runner.test()

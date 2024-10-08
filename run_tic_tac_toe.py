import os
import sys
import time
import torch
import numpy as np
from math import isnan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self, args):
        self.args = args
        # self.player1 = load_model(self.args).to(self.args.device)
        # self.player2 = load_opponent_model(self.args).to(self.args.device)
        self.player1 = load_model(self.args)
        self.player2 = load_opponent_model(self.args)
        dict_args = dict(args)
        self.simulator = Simulator(**dict_args, self.player1, self.player2)

    def train(self):
        logger.info("Start training")
        player1_win = 0.0
        player2_win = 0.0
        print_every_n = self.args.log_interval
        for i in range(1, self.args.epoch + 1):
            winner = self.simulator.play(render=self.args.render)
            if winner == 1:
                player1_win += 1
            if winner == -1:
                player2_win += 1
            if i % print_every_n == 0:
                logger.info(
                    "Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f"
                    % (i, player1_win / i, player2_win / i)
                )
            # self.player1.backup()
            # self.player2.backup()
            self.simulator.reset()
        # self.player1.save_policy()
        # self.player2.save_policy()
        logger.info("-------------------------------------------------------")

    def test(self):
        logger.info("Start testing")
        # self.player1.load_policy()
        # self.player2.load_policy()
        player1_win = 0.0
        player2_win = 0.0
        turns = self.args.test_games
        for _ in range(turns):
            winner = self.simulator.play(render=self.args.render)
            if winner == 1:
                player1_win += 1
            if winner == -1:
                player2_win += 1
            self.simulator.reset()
        logger.info(
            "%d games, player 1 winrate %.02f, player 2 winrate %.02f"
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
    runner = Runner(args)
    if args.function == 'train':
        runner.train()
    elif args.function == 'test':
        runner.test()
    else:
        print('Invalid function')

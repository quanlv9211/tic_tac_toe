from statistics import mode
from tic_tac_toe.agent.player import HumanPlayer, RandomPlayer, MiniMaxPlayer
from util import logger


def load_model(args):
    if args.your_model == "human":
        model = HumanPlayer()
    elif args.your_model == "random":
        model = RandomPlayer()
    elif args.your_model == "minimax":
        model = MiniMaxPlayer(args, args.your_symbol)
    else:
        raise Exception("Please define the model")
    logger.info("Your model is {} ".format(args.your_model))
    return model


def load_opponent_model(args):
    if args.opponent_model == "human":
        model = HumanPlayer()
    elif args.opponent_model == "random":
        model = RandomPlayer()
    elif args.opponent_model == "minimax":
        model = MiniMaxPlayer(args, args.opponent_symbol)
    else:
        raise Exception("Please define the model")
    logger.info("The opponent model is {} ".format(args.opponent_model))
    return model

from functools import partial
from ..board.board import Board
import threading
from ..agent.player import Player


class Simulator:
    def __init__(
        self,
        width: int,
        height: int,
        winstreak: int,
        player1: Player,
        player2: Player,
        **kwargs
    ):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = "X"  # X di truoc
        self.p2_symbol = "O"
        self.time_limit = None
        self.width = width
        self.height = height

        # the state is the board
        self.current_board = Board(width, height, winstreak)

    def core_reset(self):
        self.current_board = Board(self.width, self.height, self.winstreak)

    def move_fn(self, player: Player, board: Board, symbol: str):
        move = player.get_move(board)
        self.current_board.set_move(move, symbol)

    def time_contrl_move(self, player, board, symbol):
        self.current_player = (
            player,
            symbol,
        )  # use this to determine the winner if timeout happen
        mv_fn = partial(self.move_fn, player, board, symbol)
        mv_thread = threading.Thread(target=mv_fn)
        mv_thread.start()
        mv_thread.join(self.time_limit)

        if mv_thread.is_alive():
            print("time limit reached")
            raise Exception("Time limit result not implemented")

    def play(self, render=False):
        while not self.current_board.gameover():
            if render:
                self.current_board.render()

            if self.time_limit is not None:
                self.time_contrl_move(self.p1, self.current_board, "X")
            else:
                self.move_fn(self.p1, self.current_board, "X")

            if self.current_board.gameover():
                break

            if render:
                self.current_board.render()

            if self.time_limit is not None:
                self.time_contrl_move(self.p2, self.current_board, "O")
            else:
                self.move_fn(self.p2, self.current_board, "O")

        if render:
            self.current_board.render()
        if self.current_board.iswin("X"):
            if self.p1_symbol == "X":
                return 1
            else:
                return -1
        elif self.current_board.iswin("O"):
            if self.p1_symbol == "O":
                return 1
            else:
                return -1
        elif len(self.current_board.possible_moves()) == 0:
            return 0

    def reset(self):
        """
        Gym API for interacting with the environment,
        see https://github.com/Farama-Foundation/Gymnasium
        return current state of the board
        """
        self.current_board.reset()
        # bot play first
        self.move_fn(self.p1, self.current_board, "X")
        return self.get_obs(), self.current_board.available_actions()

    def get_reward(self):
        if self.current_board.gameover():
            if self.current_board.iswin("O"):
                return 1
            elif self.current_board.iswin("X"):
                return -1
        return 0

    def get_obs(self):
        return self.current_board.to_string()

    def get_done(self):
        return self.current_board.gameover()

    def step(self, action: int):
        """
        Gym API for interacting with the environment,
        see https://github.com/Farama-Foundation/Gymnasium
        """
        move = self.current_board.id2move(action)
        # agent plays as 'O'
        self.current_board.set_move(move, "O")
        if self.current_board.gameover():
            return (
                self.get_obs(),
                self.get_reward(),
                self.get_done(),
                self.current_board.available_actions(),
            )

        # bot move
        self.move_fn(self.p1, self.current_board, "X")
        return (
            self.get_obs(),
            self.get_reward(),
            self.get_done(),
            self.current_board.available_actions(),
        )

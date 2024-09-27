from ..board.board import Board
import threading


class Simulator:
    def __init__(self, args, player1, player2):
        self.args = args
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = "X"  # X di truoc
        self.p2_symbol = "O"
        self.time_limit = None

        # the state is the board
        self.current_board = Board(
            self.args.width, self.args.height, self.args.winstreak
        )

    def reset(self):
        self.current_board = Board(
            self.args.width, self.args.height, self.args.winstreak
        )

    def move_fn(self, player, board, symbol):
        move = player.get_move(board)
        self.current_board.set_move(move, symbol)

    def time_contrl_move(self, player, board, symbol):
        self.current_player = (
            player,
            symbol,
        )  # use this to determine the winner if timeout happen by handling exception
        mv_thread = threading.Thread(self.move_fn, args=(player, board, symbol))
        mv_thread.start()
        mv_thread.join(self.time_limit)

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

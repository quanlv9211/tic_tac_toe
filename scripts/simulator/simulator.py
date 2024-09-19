from ..board.board import Board

class Simulator():
    def __init__(self, args, player1, player2):
        self.args = args
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 'X'   # X di truoc
        self.p2_symbol = 'O'

        # the state is the board
        self.current_board = Board(self.args.width, self.args.height, self.args.winstreak)

    def reset(self):
        self.current_board = Board(self.args.width, self.args.height, self.args.winstreak)


    def play(self, render=False):
        while not self.current_board.gameover():
            if render:
                self.current_board.render()
            moveX = self.p1.get_move(self.current_board)
            self.current_board.set_move(moveX, 'X')

            if self.current_board.gameover(): break

            if render:
                self.current_board.render()
            moveO = self.p2.get_move(self.current_board)
            self.current_board.set_move(moveO, 'O')
        
        if render:
            self.current_board.render()
        if self.current_board.iswin('X'):
            if self.p1_symbol == 'X':
                return 1
            else:
                return -1
        elif self.current_board.iswin('O'):
            if self.p1_symbol == 'O':
                return 1
            else:
                return -1
        elif len(self.current_board.possible_moves()) == 0:
            return 0


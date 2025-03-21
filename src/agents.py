import numpy as np
import random

class ConnectFourAgent:
    """
    Een rule-based agent voor Connect Four:
    1. Probeert een winnende zet te maken.
    2. Blokkeert de tegenstander.
    3. Speelt het midden.
    4. Anders: random geldige zet.
    """

    def __init__(self, player):
        self.player = player

    def select_action(self, board):
        board = np.array(board)
        
        # 1. Winnende zet
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                if self.check_win(temp_board, self.player):
                    return col

        # 2. Blokkeer tegenstander
        opponent = 3 - self.player
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, opponent)
                if self.check_win(temp_board, opponent):
                    return col

        # 3. Midden
        if self.is_valid_move(board, 3):
            return 3

        # 4. Random
        valid_moves = [c for c in range(board.shape[1]) if self.is_valid_move(board, c)]
        return random.choice(valid_moves)

    def is_valid_move(self, board, col):
        return board[0][col] == 0

    def make_move(self, board, col, player):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 0:
                board[row][col] = player
                break

    def check_win(self, board, player):
        # (Horizontaal, verticaal, diagonalen check)
        # ... zie je code van eerder ...
        # Als je code al goed werkt, laat dit zo.
        # -----------------------------------------
        # Horizontaal
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True

        # Verticaal
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                if all(board[row + i][col] == player for i in range(4)):
                    return True

        # Diagonaal linksboven -> rechtsonder
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True

        # Diagonaal rechtsboven -> linksonder
        for row in range(3, board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row - i][col + i] == player for i in range(4)):
                    return True

        return False


class RandomAgent:
    """ Eenvoudige baseline-agent die willekeurig zet. """

    def __init__(self, player):
        self.player = player

    def select_action(self, board):
        valid_moves = [c for c in range(board.shape[1]) if board[0][c] == 0]
        return random.choice(valid_moves)
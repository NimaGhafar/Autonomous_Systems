import numpy as np
import math
import random
import time

class MinMaxAgent:
    """
    Deze agent doet het volgende om Connect Four te spelen:
    - Minimax met alpha-beta pruning
    - Iteratieve verdieping
    - Transpositietabel caching
    - Verbeterde move ordering
    - Expliciete check op onmiddellijke winnende zetten
    """
    
    def __init__(self, player, max_depth=12, time_limit=5.0):
        self.player = player
        self.max_depth = max_depth      # Maximale zoekdiepte
        self.time_limit = time_limit    # Tijdslimiet per zet in seconden
        self.start_time = None
        self.transposition_table = {}

    def select_action(self, board):
        self.start_time = time.time()
        self.transposition_table = {}  # Reset de transpositietabel voor elke zet
        valid_moves = [c for c in range(board.shape[1]) if self.is_valid_move(board, c)]
        
        # 1. Directe check: als er een zet is die direct leidt tot 4 op een rij, speel die!
        for col in valid_moves:
            temp_board = board.copy()
            self.make_move(temp_board, col, self.player)
            if self.check_win(temp_board, self.player):
                return col

        best_move = valid_moves[0]
        best_score = -math.inf

        # Zetvolgorde op basis van afstand tot het centrum 
        valid_moves = sorted(valid_moves, key=lambda col: abs(col - board.shape[1]//2))
        
        # 2. Iteratieve verdieping: zoek eerst met een geringe diepte, verhoog dan de diepte
        for depth in range(1, self.max_depth + 1):
            for col in valid_moves:
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                score = self.minimax(temp_board, depth - 1, -math.inf, math.inf, False)
                if score > best_score:
                    best_score = score
                    best_move = col
                if time.time() - self.start_time > self.time_limit:
                    break
            if time.time() - self.start_time > self.time_limit:
                break

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_board(board)

        board_key = self.board_to_tuple(board)
        if board_key in self.transposition_table:
            stored_depth, stored_score = self.transposition_table[board_key]
            if stored_depth >= depth:
                return stored_score

        valid_moves = [c for c in range(board.shape[1]) if self.is_valid_move(board, c)]
        is_terminal = self.is_terminal_node(board)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_win(board, self.player):
                    return 1000000
                elif self.check_win(board, 3 - self.player):
                    return -1000000
                else:
                    return 0  
            else:
                return self.evaluate_board(board)
        
        if maximizingPlayer:
            value = -math.inf
            valid_moves = sorted(valid_moves, key=lambda col: abs(col - board.shape[1]//2))
            for col in valid_moves:
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                value = max(value, self.minimax(temp_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self.transposition_table[board_key] = (depth, value)
            return value
        else:
            value = math.inf
            valid_moves = sorted(valid_moves, key=lambda col: abs(col - board.shape[1]//2))
            for col in valid_moves:
                temp_board = board.copy()
                self.make_move(temp_board, col, 3 - self.player)
                value = min(value, self.minimax(temp_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self.transposition_table[board_key] = (depth, value)
            return value

    def board_to_tuple(self, board):
        """Zet het bord om naar een tuple zodat je het kunt gebruiken als sleutel in een soort geheugenlijst (de transpositietabel)."""
        return tuple(map(tuple, board))

    def is_terminal_node(self, board):
        return (self.check_win(board, self.player) or 
                self.check_win(board, 3 - self.player) or 
                all(board[0][c] != 0 for c in range(board.shape[1])))

    def evaluate_board(self, board):
        # Evaluatie van het bord uit het perspectief van self.player.
        score = 0
        score += self.score_position(board, self.player)
        score -= self.score_position(board, 3 - self.player)
        return score

    def score_position(self, board, player):
        score = 0
        # Voorkeur voor het centrum: tel het aantal schijven in de centrale kolom.
        center_array = list(board[:, board.shape[1]//2])
        center_count = center_array.count(player)
        score += center_count * 3

        # Horizontale windows
        for row in range(board.shape[0]):
            row_array = list(board[row, :])
            for col in range(board.shape[1] - 3):
                window = row_array[col:col+4]
                score += self.evaluate_window(window, player)
        
        # Verticale windows
        for col in range(board.shape[1]):
            col_array = list(board[:, col])
            for row in range(board.shape[0] - 3):
                window = col_array[row:row+4]
                score += self.evaluate_window(window, player)
        
        # Diagonale windows (positieve helling)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        # Diagonale windows (negatieve helling)
        for row in range(3, board.shape[0]):
            for col in range(board.shape[1] - 3):
                window = [board[row-i][col+i] for i in range(4)]
                score += self.evaluate_window(window, player)
        
        return score

    def evaluate_window(self, window, player):
        score = 0
        opponent = 3 - player
        
        if window.count(player) == 4:
            score += 10000
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 100
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 10

        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 120 
        
        return score

    def is_valid_move(self, board, col):
        return board[0][col] == 0

    def make_move(self, board, col, player):
        for row in range(board.shape[0]-1, -1, -1):
            if board[row][col] == 0:
                board[row][col] = player
                break

    def check_win(self, board, player):
        # Horizontale check
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row][col+i] == player for i in range(4)):
                    return True

        # Verticale check
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                if all(board[row+i][col] == player for i in range(4)):
                    return True

        # Diagonaal (linksboven naar rechtsonder)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                if all(board[row+i][col+i] == player for i in range(4)):
                    return True

        # Diagonaal (rechtsboven naar linksonder)
        for row in range(3, board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row-i][col+i] == player for i in range(4)):
                    return True

        return False


class RandomAgent:
    """Eenvoudige baseline-agent die een willekeurige zet kiest."""
    
    def __init__(self, player):
        self.player = player

    def select_action(self, board):
        valid_moves = [c for c in range(board.shape[1]) if board[0][c] == 0]
        return random.choice(valid_moves)
    

class DoubleMoveAgent:
    """
    Deze agent doet het volgende om Connect Four te spelen:
    1. Probeer een dubbele zet te maken.
    2. Blokkeer de tegenstander.
    3. Anders: random geldige zet.
    4. Probeer weer een dubbele zet te maken.
    """
    def __init__(self, player):
        self.player = player

    def select_action(self, board):
        # Probeer dubbele zet te creëren in twee opeenvolgende beurten
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                # Controleer of de volgende zet in dezelfde kolom mogelijk is
                if self.is_valid_move(temp_board, col):
                    temp_board_next = temp_board.copy()
                    self.make_move(temp_board_next, col, self.player)
                    if self.check_win(temp_board_next, self.player):
                        return col

        # Blokkeer tegenstander
        opponent = 3 - self.player
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, opponent)
                if self.count_winning_moves(temp_board, opponent) > 0:
                    return col

        # Probeer weer dubbele zet te creëren
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                if self.count_winning_moves(temp_board, self.player) > 1:
                    return col

        # Anders willekeurige zet
        valid_moves = [c for c in range(board.shape[1]) if self.is_valid_move(board, c)]
        return random.choice(valid_moves)

    def is_valid_move(self, board, col):
        return board[0][col] == 0

    def make_move(self, board, col, player):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 0:
                board[row][col] = player
                break

    def count_winning_moves(self, board, player):
        count = 0
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, player)
                if self.check_win(temp_board, player):
                    count += 1
        return count

    def check_win(self, board, player):
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                if all(board[row + i][col] == player for i in range(4)):
                    return True
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True
            for col in range(3, board.shape[1]):
                if all(board[row + i][col - i] == player for i in range(4)):
                    return True
        return False

class GreedyAgent:
    """
    Deze agent doet het volgende om Connect Four te spelen:
    1. Kijkt naar de score na één zet.
    """
    def __init__(self, player):
        self.player = player

    def select_action(self, board):
        valid_moves = [c for c in range(board.shape[1]) if board[0][c] == 0]
        best_score = -float('inf')
        best_move = valid_moves[0]

        for col in valid_moves:
            temp_board = board.copy()
            self.make_move(temp_board, col, self.player)
            score = self.evaluate_board(temp_board)
            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def make_move(self, board, col, player):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 0:
                board[row][col] = player
                break

    def evaluate_board(self, board):
        player_count = np.count_nonzero(board == self.player)
        opponent_count = np.count_nonzero(board == (3 - self.player))
        return player_count - opponent_count
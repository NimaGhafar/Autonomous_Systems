import numpy as np
import random
from pettingzoo.classic import connect_four_v3

class ConnectFourAgent:
    def __init__(self, player):
        self.player = player
    
    def select_action(self, board):
        board = np.array(board)

        # 1. Winnende zet spelen als dat mogelijk is
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, self.player)
                if self.check_win(temp_board, self.player):
                    return col
        
        # 2. Voorkomen dat tegenstander wint
        opponent = 3 - self.player
        for col in range(board.shape[1]):
            if self.is_valid_move(board, col):
                temp_board = board.copy()
                self.make_move(temp_board, col, opponent)
                if self.check_win(temp_board, opponent):
                    return col
        
        # 3. Strategische zet maken (midden pakken)
        if self.is_valid_move(board, 3):  
            return 3
        
        # 4. Willekeurige geldige zet maken
        valid_moves = [col for col in range(board.shape[1]) if self.is_valid_move(board, col)]
        return random.choice(valid_moves)

    def is_valid_move(self, board, col):
        return board[0][col] == 0
    
    def make_move(self, board, col, player):
        for row in reversed(range(board.shape[0])):
            if board[row][col] == 0:
                board[row][col] = player
                break
    
    def check_win(self, board, player):
        # Horizontale controle
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True
        
        # Verticale controle
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                if all(board[row + i][col] == player for i in range(4)):
                    return True
        
        # Diagonale controle (van linksboven naar rechtsonder)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True
        
        # Diagonale controle (van rechtsboven naar linksonder)
        for row in range(3, board.shape[0]):
            for col in range(board.shape[1] - 3):
                if all(board[row - i][col + i] == player for i in range(4)):
                    return True
        
        return False

# Test de agent
if __name__ == "__main__":
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    agent = ConnectFourAgent(player=1)

    done = False

    while not done:
        env.render()

        # ✅ Bord correct ophalen uit de dictionary
        observation = env.observe(env.agent_selection)
        print("Observation:", observation)  

        # Alleen het bord pakken uit de observatie
        board = np.array(observation['observation'][:, :, 0])

        # Controleer welke speler aan de beurt is
        current_player = 1 if env.agent_selection == "player_0" else 2
        
        if current_player == agent.player:
            action = agent.select_action(board)
            env.step(action)
        else:
            action = int(input("Kies een kolom (0-6): "))
            env.step(action)
        
        # Controleer of het spel is afgelopen
        done = env.terminations["player_0"] or env.terminations["player_1"]
    
    env.render()
    winner = 1 if env.rewards["player_0"] == 1 else (2 if env.rewards["player_1"] == 1 else 0)
    if winner:
        print(f"Speler {winner} heeft gewonnen!")
    else:
        print("Gelijkspel!")

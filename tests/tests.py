import unittest
import numpy as np
from src.agents import ConnectFourAgent

class TestConnectFourAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ConnectFourAgent(player=1)

    def test_winning_move(self):
        # Maak een bord waar speler 1 (agent) bijna wint
        board = np.zeros((6, 7), dtype=int)
        # Plaats drie schijven op rij [5,3], [5,4], [5,5]
        board[5,3] = 1
        board[5,4] = 1
        board[5,5] = 1
        # De agent zou nu kolom 6 moeten kiezen (om 4 op een rij te maken)
        move = self.agent.select_action(board)
        self.assertIn(move, [2, 6], f"Verwachtte een winnende zet (2 of 6), maar agent koos kolom {move}")

if __name__ == "__main__":  
    unittest.main()
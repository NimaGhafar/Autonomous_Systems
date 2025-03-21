import numpy as np
from pettingzoo.classic import connect_four_v3
from agents import ConnectFourAgent  # importeer je agent (of baseline)

def interactive_game():
    """
    Speel een interactief potje (mens vs. agent) met visuele rendering.
    """
    env = connect_four_v3.env(render_mode="human")
    env.reset()
    agent = ConnectFourAgent(player=1)  # Speler 1 = rule-based agent

    done = False
    while not done:
        env.render()  # Toon het bord
        
        observation = env.observe(env.agent_selection)
        board = observation['observation'][:, :, 0]
        current_player = 1 if env.agent_selection == "player_0" else 2

        if current_player == agent.player:
            # Agent zet
            action = agent.select_action(board)
            env.step(action)
        else:
            # Menselijke speler
            user_input = input("Kies een kolom (0-6): ")
            try:
                col = int(user_input)
            except ValueError:
                print("Ongeldige invoer, probeer opnieuw!")
                continue
            env.step(col)
        
        done = any(env.terminations.values()) or any(env.truncations.values())
    
    # Eindsituatie
    env.render()
    reward_p0 = env.rewards["player_0"]
    reward_p1 = env.rewards["player_1"]
    if reward_p0 == 1:
        winner = 1
    elif reward_p1 == 1:
        winner = 2
    else:
        winner = 0

    if winner:
        print(f"Speler {winner} heeft gewonnen!")
    else:
        print("Gelijkspel!")


if __name__ == "__main__":
    # Als je in de terminal/command line `python main.py` doet, start het interactieve spel
    interactive_game()
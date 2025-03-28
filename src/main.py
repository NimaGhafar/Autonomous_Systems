import numpy as np
from pettingzoo.classic import connect_four_v3
from agents import MinMaxAgent, RandomAgent, ConnectFourAgent, GreedyAgent

def choose_agent(player=1):
    """
    Laat de gebruiker kiezen welke agent hij wil gebruiken.
    """
    print("Tegen welke agent wil je spelen?")
    print("1) MinMaxAgent")
    print("2) RandomAgent")
    print("3) ConnectFourAgent (Rule-based)")
    print("4) GreedyAgent")

    choice = input("Voer je keuze in (1-4): ")
    if choice == "1":
        return MinMaxAgent(player=player)
    elif choice == "2":
        return RandomAgent(player=player)
    elif choice == "3":
        return ConnectFourAgent(player=player)
    elif choice == "4":
        return GreedyAgent(player=player)
    else:
        print("Ongeldige keuze. Er wordt standaard voor ConnectFourAgent gekozen.")
        return ConnectFourAgent(player=player)


def interactive_game():
    """
    Speel een interactief potje (mens vs. agent) met visuele rendering.
    """
    agent = choose_agent(player=1) 

    env = connect_four_v3.env(render_mode="human")
    env.reset()

    done = False
    while not done:
        env.render()
        
        observation = env.observe(env.agent_selection)
        board = observation['observation'][:, :, 0]
        current_player = 1 if env.agent_selection == "player_0" else 2

        if current_player == agent.player:
            action = agent.select_action(board)
            env.step(action)
        else:
            user_input = input("Kies een kolom (0-6): ")
            try:
                col = int(user_input)
            except ValueError:
                print("Ongeldige invoer, probeer opnieuw!")
                continue
            env.step(col)
        
        done = any(env.terminations.values()) or any(env.truncations.values())
    
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
    interactive_game()
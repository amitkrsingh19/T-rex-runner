import csv

from env.dino_env import DinoEnv
from env.game_setup import GameControl
from agent.dqn import DQNAgent

import statistics


# ANSI Color Constants
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

def main():

    game = GameControl()
    game.start_game()
    
    env = DinoEnv()

    #obs, info = env.reset()
    num_actions = env.action_space.n

    agent = DQNAgent(num_actions=num_actions)

    num_episodes = 500
    save_checkpoint_frequency = 50
    
#    print("\n" + f"{GREEN}={RESET}"*45)
#    print(f"{RED}  Starting DQN Training on Chrome Dino Game{RESET}")
#    print(f"{GREEN}={RESET}"*45 + "\n")   

    with open("logs/training_log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "score", "steps", "epsilon", "avg_loss"])

    for episode in range(num_episodes):

        state, info = env.reset()
        done = False
        total_reward = 0
        total_steps = 0
        total_loss = []

        while not done:
            try:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.remember(state, action, reward, next_state, done)

                loss = agent.train()
                if loss:
                    total_loss.append(loss)

                agent.decay_epsilon()
                
                state = next_state
                total_reward += reward
                total_steps += 1

#            if done:
#                final_score = info.get('score', 0)#

#                print(f"{BLUE} Episode:{YELLOW} {episode + 1}/{num_episodes}{RESET}")
#                print(f"{BLUE} Steps Survived:{YELLOW} {episode_step_count}{RESET}")
#                print(f"{BLUE} Final Game Score:{YELLOW} {final_score}{RESET}")
#                print(f"{BLUE} Total Accrued Reward:{YELLOW} {total_reward}{RESET}")
#                print(f"{BLUE} Current Exploration Epsilon:{YELLOW} {agent.epsilon:.4f}{RESET}")
#                print("-" * 45)

        # Periodic model tracking save sequences
            except Exception as e:
                print(f"[ERROR] Episode {episode + 1} crashed : {e}")
                done = True

        avg_loss = statistics.mean(total_loss) if total_loss else 0.0

        with open("logs/training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([episode + 1, total_reward, info.get("score", 0), total_steps, agent.epsilon, avg_loss])

        if (episode + 1) % save_checkpoint_frequency == 0:
            agent.update_target_network()

        if (episode + 1) % save_checkpoint_frequency == 0:
            model_save_path = f"checkpoints/dino_dqn_ep{episode + 1}.keras"
            agent.save_model(model_save_path)
            print(f"{RED}[SAVED]{BLUE} Saved target model to {model_save_path}\n{RESET}")

        print(f"{RED}Episode {YELLOW}{episode}   {GREEN}reward {YELLOW}{total_reward}   {BLUE}epsilon {YELLOW}{agent.epsilon:.3f} {CYAN}lOSS  {YELLOW}{avg_loss:.4f}{RESET}")

    env.close()

if __name__ == "__main__":
    main()
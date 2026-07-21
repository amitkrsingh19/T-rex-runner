from env.dino_env import DinoEnv
from env.game_setup import GameControl
from agent.dqn import DQNAgent


# ANSI Color Constants
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def main():

    game = GameControl()
    game.start_game()
    
    env = DinoEnv()

    #obs, info = env.reset()
    state_shape = env.observation_space.shape 
    num_actions = env.action_space.n

    agent = DQNAgent(state_shape=state_shape, num_actions=num_actions)

    num_episodes = 50
    save_checkpoint_frequency = 10
    
    print("\n" + f"{GREEN}={RESET}"*40)
    print(f"{RED}  Starting DQN Training on Chrome Dino Game{RESET}")
    print(f"{GREEN}={RESET}"*45 + "\n")   
    
    for episode in range(num_episodes):

        state, info = env.reset()

        agent.memory.clear()

        done = False
        total_reward = 0
        episode_step_count = 0

        while not done:
            env.render()

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)

            agent.train()

            state = next_state
            total_reward += reward
            episode_step_count +=1 

            if done:
                final_score = info.get('score', 0)

                print(f"{BLUE} Episode:{YELLOW} {episode + 1}/{num_episodes}{RESET}")
                print(f"{BLUE} Steps Survived:{YELLOW} {episode_step_count}{RESET}")
                print(f"{BLUE} Final Game Score:{YELLOW} {final_score}{RESET}")
                print(f"{BLUE} Total Accrued Reward:{YELLOW} {total_reward}{RESET}")
                print(f"{BLUE} Current Exploration Epsilon:{YELLOW} {agent.epsilon:.4f}{RESET}")
                print("-" * 45)

        # Periodic model tracking save sequences
        if (episode + 1) % save_checkpoint_frequency == 0:
            model_save_path = f"checkpoints/dino_dqn_ep{episode + 1}.h5"
            agent.save_model(model_save_path)
            print(f"{RED}[SAVED]{BLUE} Saved target model to {model_save_path}\n{RESET}")   

    env.close()

if __name__ == "__main__":
    main()
from env.dino_env import DinoEnv
from env.game_setup import GameControl

def main():

    game = GameControl()
    game.start_game()
    
    env = DinoEnv()

    #obs, info = env.reset()

    for episode in range(5):
        print()
        print(f"Starting Episode {episode + 1}")
        print("---" * 20)
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()

            action = env.action_space.sample() 

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            if done:
                print(f"Finished Episode with {info}")
                print(f"Total Reward for this Episode is {total_reward}")
                print("---" * 20)
    
    env.close()
if __name__ == "__main__":
    main()
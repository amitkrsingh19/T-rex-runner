import time

import numpy as np
import tensorflow as tf

from env.dino_env import DinoEnv
from env.game_setup import GameControl


MODEL_PATH = "checkpoints/best_model.keras"
NUM_GAMES = 3


def evaluate(agent_type="dqn", model_path=None, num_games=None):
    if model_path is None:
        model_path = MODEL_PATH
    if num_games is None:
        num_games = NUM_GAMES

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    game = GameControl()
    game.start_game()

    env = DinoEnv()

    for episode in range(num_games):
        state, info = env.reset()
        done = False
        total_steps = 0
        total_reward = 0

        print(f"\n--- Starting Game {episode + 1} ---")

        while not done:
            # Prepare state exactly as in training
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0
            state_tensor = tf.expand_dims(state_tensor, axis=0)

            # Best action, no exploration
            q_values = model(state_tensor, training=False)
            action = int(np.argmax(q_values.numpy()[0]))

            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_steps += 1
            total_reward += reward

            # Render agent's vision
            env.render()

        final_score = info.get('score', 0)
        print(f"Game Over! Steps Survived: {total_steps} | Score: {final_score}")
        time.sleep(1)

    env.close()
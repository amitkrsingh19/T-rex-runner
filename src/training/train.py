import os
import csv
import statistics
import traceback
import time

import numpy as np
import tensorflow as tf

from env.dino_env import DinoEnv
from env.game_setup import GameControl
from agent.dqn import DQNAgent
from config import BATCH_SIZE


def check_gpu():
    """Proper GPU detection"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                print(f"GPU Found: {gpu}")
            print(f"Number of GPUs: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")


def train(agent_type="dqn", episodes=500, writer=None, resume_from=None):
    # --- BUILD ---
    check_gpu()

    game = GameControl()
    game.start_game()

    env = DinoEnv(render_mode="human")
    num_actions = env.action_space.n

    # Agent factory
    AGENTS = {
        "dqn": DQNAgent,
        # "double_dqn": DoubleDQNAgent,
        # "dueling_dqn": DuelingDQNAgent,
        # "expected_sarsa": ExpectedSARSAAgent,
    }
    agent_class = AGENTS.get(agent_type, DQNAgent)
    agent = agent_class(num_actions=num_actions)

    # Load pretrained weights if resume path provided
    if resume_from is not None:
        try:
            agent.policy_network.model.load_weights(resume_from)
            agent.update_target_network()
            print(f"Loaded pretrained weights from {resume_from}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        # Try default checkpoint
        try:
            agent.policy_network.model.load_weights('checkpoints/best_model3.keras')
            agent.update_target_network()
            print("Loaded pretrained weights successfully!")
        except Exception:
            print("No pretrained weights found, starting fresh.")

    # Training parameters
    num_episodes = episodes
    save_checkpoint_frequency = 100
    target_update_frequency = 10
    train_every = 4
    best_score = 0

    # Create log file
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    with open("logs/training_log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "score", "steps", "epsilon", "avg_loss"])

    global_step = 0

    # --- RUN ---
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        total_loss = []
        step_counter = 0

        while not done:
            try:
                env.render()

                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.remember(state, action, reward, next_state, done)

                # Periodic training during gameplay
                step_counter += 1
                global_step += 1

                if step_counter % train_every == 0:
                    loss = agent.train()
                    if loss is not None:
                        total_loss.append(loss)

                # Per-step epsilon decay
                agent.decay_epsilon_step()

                # TensorBoard: per-step logging
                if writer is not None and len(total_loss) > 0:
                    with writer.as_default():
                      tf.summary.scalar("train/loss", total_loss[-1], step=global_step)
                      tf.summary.scalar("train/epsilon", agent.epsilon, step=global_step)

                state = next_state
                total_reward += reward
                episode_steps += 1

            except Exception as e:
                print(f"[ERROR] Episode {episode + 1} crashed : {e}")
                traceback.print_exc()
                done = True

        # Calculate average loss
        avg_loss = statistics.mean(total_loss) if total_loss else 0.0

        # End-of-episode batch training
        if len(agent.memory) >= BATCH_SIZE:
            num_batches = min(episode_steps // 2, 500)
            for _ in range(num_batches):
                loss = agent.train()
                if loss is not None:
                    total_loss.append(loss)

        # Per-episode epsilon decay
        agent.decay_epsilon()

        # Get final score
        final_score = int(info.get('score', 0))

        if final_score > best_score:
            best_score = final_score
            agent.save_model('checkpoints/best_model.keras')

        # Log to CSV
        with open("logs/training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                episode + 1,
                round(total_reward, 3),
                final_score,
                episode_steps,
                agent.epsilon,
                avg_loss
            ])

        # TensorBoard: per-episode logging
        if writer is not None:
            with writer.as_default():
              tf.summary.scalar("episode/reward", total_reward, step=episode)
              tf.summary.scalar("episode/score", final_score, step=episode)
              tf.summary.scalar("episode/length", episode_steps, step=episode)
            if total_loss:
                tf.summary.scalar("episode/avg_loss", avg_loss, step=episode)

        # Target network update
        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        # Checkpoint save
        if (episode + 1) % save_checkpoint_frequency == 0:
            model_save_path = f"checkpoints/dino_dqn_ep{episode + 1}.keras"
            agent.save_model(model_save_path)

    env.close()
    print("\nTraining Complete!")
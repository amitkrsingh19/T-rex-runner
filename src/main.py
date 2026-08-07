import argparse
from datetime import datetime
import tensorflow as tf
from training.train import train
from training.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Chrome Dino DQN")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--agent", choices=["dqn", "double_dqn", "dueling_dqn", "expected_sarsa"], default="dqn")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.mode == "train":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = tf.summary.create_file_writer(f"logs/{args.agent}_{run_id}")
        train(
            agent_type=args.agent,
            episodes=args.episodes,
            writer=writer,
            resume_from=args.resume,
        )
        writer.close()
    else:
        evaluate(agent_type=args.agent)


if __name__ == "__main__":
    main()
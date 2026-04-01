import argparse
import sys

from src.config import Config


def main():
    parser = argparse.ArgumentParser(prog="coding-llm")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download", help="Download & filter datasets")
    sub.add_parser("tokenise", help="Train tokeniser & encode shards")
    sub.add_parser("embeddings", help="Download GloVe & align to vocab")
    sub.add_parser("train", help="Train the model")
    sub.add_parser("evaluate", help="Evaluate a checkpoint")
    sub.add_parser("generate", help="Generate code completions")

    args, remaining = parser.parse_known_args()

    # Load configuration
    config = Config()

    if args.command == "download":
        from src.dataset_procesing.download import main as run
    elif args.command == "tokenise":
        from src.dataset_procesing.tokeniser import main as run
    elif args.command == "embeddings":
        from src.dataset_procesing.embeddings import main as run
    elif args.command == "train":
        from src.training_and_evaluation.train import main as run
    elif args.command == "evaluate":
        from src.training_and_evaluation.evaluate import main as run
    elif args.command == "generate":
        from src.training_and_evaluation.generate import main as run
    else:
        parser.print_help()
        sys.exit(0)

    # Pass the loaded configuration to the respective main functions
    sys.argv = [sys.argv[0]] + remaining
    run(config)


if __name__ == "__main__":
    main()

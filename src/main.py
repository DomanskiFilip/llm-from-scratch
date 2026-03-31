import argparse
import sys


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

    if args.command == "download":
        from dataset_procesing.download import main as run
    elif args.command == "tokenise":
        from dataset_procesing.tokeniser import main as run
    elif args.command == "embeddings":
        from dataset_procesing.embedings import main as run
    elif args.command == "train":
        from training_and_evaluation.train import main as run
    elif args.command == "evaluate":
        from training_and_evaluation.evaluate import main as run
    elif args.command == "generate":
        from training_and_evaluation.generate import main as run
    else:
        parser.print_help()
        sys.exit(0)

    sys.argv = [sys.argv[0]] + remaining
    run()


if __name__ == "__main__":
    main()

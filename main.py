import argparse
from typing import List, Optional

from fm_gas.cli_args import add_generate_args, add_train_args
from fm_gas.generate import generate
from fm_gas.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conditional flow-matching for 3D gas trajectories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train autoregressive flow-matching model")
    add_train_args(train_parser)

    gen_parser = subparsers.add_parser("generate", help="Generate sensor trajectories autoregressively")
    add_generate_args(gen_parser)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

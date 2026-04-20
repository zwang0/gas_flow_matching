import argparse
from typing import List, Optional

from fm_gas.cli_args import add_common_io, add_constraint_args
from fm_gas.generate import generate
from fm_gas.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conditional flow-matching for 3D gas trajectories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train conditional flow-matching model")
    add_common_io(train_parser)
    add_constraint_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--checkpoint", type=str, default="checkpoints/flow_matcher_09_18_09_50sccm.pt")

    gen_parser = subparsers.add_parser("generate", help="Generate concentration trajectories")
    add_common_io(gen_parser)
    add_constraint_args(gen_parser)
    gen_parser.add_argument("--checkpoint", type=str, required=True)
    gen_parser.add_argument("--positions-csv", type=str, default="", help="Optional CSV with x_m,y_m,z_m")
    gen_parser.add_argument("--num-steps", type=int, default=200, help="Euler integration steps")
    gen_parser.add_argument("--output-csv", type=str, default="outputs/generated_trajectories.csv")

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

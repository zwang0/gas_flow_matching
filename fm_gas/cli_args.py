import argparse


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing *_binned.csv and *_surface_averages.csv files.",
    )
    p.add_argument(
        "--sensor-coords-csv",
        type=str,
        default="data/sensor_coords.csv",
        help="CSV with sensor positions (columns: sensor,x,y,z).",
    )
    p.add_argument(
        "--inlet-outlet-coords-csv",
        type=str,
        default="data/inlet_outlet_coords.csv",
        help="CSV with inlet/outlet coordinates.",
    )
    p.add_argument("--history-k", type=int, default=1, help="History window size.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-traj", type=int, default=0, help="Limit number of trajectories (0 uses all).")
    p.add_argument("--prefix-loss-weight", type=float, default=1.0)
    p.add_argument("--prefix-batch-size", type=int, default=0, help="Batch size for prefix initializer (0 uses batch-size).")
    p.add_argument("--save-every", type=int, default=0, help="Save a checkpoint every k epochs (0 disables).")
    p.add_argument("--checkpoint", type=str, default="checkpoints/flow_matcher.pt")


def add_generate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sensor-coords-csv",
        type=str,
        default="data/sensor_coords.csv",
        help="CSV with sensor positions (columns: sensor,x,y,z).",
    )
    p.add_argument(
        "--inlet-outlet-coords-csv",
        type=str,
        default="data/inlet_outlet_coords.csv",
        help="CSV with inlet/outlet coordinates.",
    )
    p.add_argument(
        "--binned-csv",
        type=str,
        required=True,
        help="Binned CSV with x_m,y_m,z_m and c_{time} columns.",
    )
    p.add_argument(
        "--surface-avg-csv",
        type=str,
        required=True,
        help="Surface averages CSV providing sensor time series.",
    )
    p.add_argument("--num-steps", type=int, default=50, help="Euler integration steps per prediction.")
    p.add_argument("--cold-start", action="store_true", help="Use learned prefix initializer for t < k.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-csv", type=str, default="outputs/generated_surface_trajectory.csv")

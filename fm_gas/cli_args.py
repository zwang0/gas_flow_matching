import argparse


def add_common_io(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--traj-csv",
        type=str,
        default="data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_concentration.csv",
        help="Trajectory CSV with x_m,y_m,z_m and c_{time} columns.",
    )
    p.add_argument(
        "--sensor-csv",
        type=str,
        default="data/Trajectories_3D/Gas_3D_sim09_18_09_50sccm_table1.csv",
        help="Sensor time series CSV.",
    )
    p.add_argument(
        "--sensor-coords-csv",
        type=str,
        default="data/sensor_coords.csv",
        help="Sensor coordinates CSV.",
    )
    p.add_argument(
        "--inlet-outlet-coords-csv",
        type=str,
        default="data/inlet_outlet_coords.csv",
        help="Inlet/outlet coordinates CSV.",
    )
    p.add_argument("--inlet-id", type=int, default=18, help="Inlet index in inlet_outlet_coords.csv")
    p.add_argument("--outlet-id", type=int, default=9, help="Outlet index in inlet_outlet_coords.csv")
    p.add_argument("--flow-sccm", type=float, default=50.0, help="Flow speed in sccm")
    p.add_argument(
        "--sensor-map-k",
        type=int,
        default=1,
        help="Number of nearest spatial points used to map each sensor location.",
    )


def add_constraint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--constraint-mode",
        type=str,
        choices=["none", "soft", "hard", "hybrid"],
        default="hybrid",
        help="Constraint behavior: soft/hybrid adds train-time sensor loss, hard/hybrid adds sampling projection.",
    )
    p.add_argument(
        "--lambda-sensor",
        type=float,
        default=1.0,
        help="Weight for train-time sensor consistency loss.",
    )
    p.add_argument(
        "--lambda-sensor-warmup-epochs",
        type=int,
        default=30,
        help="Warmup epochs for gradually enabling sensor consistency loss.",
    )
    p.add_argument(
        "--projection-alpha",
        type=float,
        default=1.0,
        help="Projection strength in sampling for hard/hybrid constraints (0 to 1).",
    )
    p.add_argument(
        "--projection-every",
        type=int,
        default=1,
        help="Apply projection every N Euler steps during sampling.",
    )

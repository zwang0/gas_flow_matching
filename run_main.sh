#!/bin/bash
#SBATCH --job-name=fm_gas_job
#SBATCH --output=/home/wjq8vw/projects/phycis_flow_matching/output/logs/slurm_%x_%j.log
#SBATCH --error=/home/wjq8vw/projects/phycis_flow_matching/output/logs/slurm_%x_%j.err
#SBATCH --time=24:00:00             # Adjust time as needed (HH:MM:SS)
#SBATCH --partition=gpu              # Specify your cluster's GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU (assuming this is a PyTorch/DL job)
#SBATCH --cpus-per-task=4            # Number of CPU cores
#SBATCH --mem=32G                    # Total memory requested
#SBATCH --account=raiselab

# 1. Navigate to the project directory
cd /home/wjq8vw/projects/phycis_flow_matching/

# 2. Load necessary modules or activate your virtual environment
# Uncomment and adjust the lines below based on your cluster's setup:
# module load python/3.9
# module load cuda/11.8
# PYTHON_BIN=/home/wjq8vw/minifogre3/envs/PDM/bin/python

eval "$(conda shell.bash hook)"
conda activate PDM

# 3. Execute the script
# Below is the command to run the "train" subparser. 
# You can adjust the arguments or swap it with the "generate" command below it.

echo "Starting training job..."

python main.py train \
    --data-dir data \
    --sensor-coords-csv data/sensor_coords.csv \
    --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
    --history-k 1 \
    --epochs 3 \
    --batch-size 256 \
    --lr 2e-4 \
    --hidden-dim 128

echo "Job finished successfully."

# ------------------------------------------------------------------------------
# ALTERNATIVE: If you want to run the "generate" command instead, comment out 
# the python block above and uncomment the block below:
# ------------------------------------------------------------------------------
# echo "Starting generation job..."
# python main.py generate \
#     --checkpoint "checkpoints/flow_matcher.pt" \
#     --sensor-coords-csv data/sensor_coords.csv \
#     --inlet-outlet-coords-csv data/inlet_outlet_coords.csv \
#     --init-surface-csv data/Gas_3D_sim08_19_09_50sccm_surface_averages.csv \
#     --trajectory-length 100 \
#     --num-steps 50 \
#     --output-csv "outputs/generated_surface_trajectory.csv"
#!/bin/bash

# Default values
EPISODES=10000
BATCH_SIZE=256
GRID_SIZE=20
POWER_DURATION=10
INDEPENDENT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --episodes) EPISODES="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --grid_size) GRID_SIZE="$2"; shift ;;
        --power_duration) POWER_DURATION="$2"; shift ;;
        --independent) INDEPENDENT="--independent" ;;
        -h|--help) 
            echo "Usage: ./run.sh [options]"
            echo "Options:"
            echo "  --episodes <int>          Number of training episodes (default: 10000)"
            echo "  --batch_size <int>        Batch size for replay buffer (default: 1024)"
            echo "  --grid_size <int>         Size of the pac-man grid (default: 20)"
            echo "  --power_duration <int>    Power pill active duration (default: 10)"
            echo "  --independent             Run Independent DDPG instead of Centralized MADDPG"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================="
echo " Starting MADDPG Training Pipeline"
echo "========================================="
echo "Episodes:       $EPISODES"
echo "Grid Size:      $GRID_SIZE"
echo "Power Duration: $POWER_DURATION"
echo "Algorithm:      $(if [ -n "$INDEPENDENT" ]; then echo "Independent DDPG"; else echo "MADDPG"; fi)"
echo "========================================="

# 1. Run the Training Script
python train.py --episodes $EPISODES --batch_size $BATCH_SIZE --grid_size $GRID_SIZE --power_duration $POWER_DURATION $INDEPENDENT

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo ""
echo "========================================="
echo " Training Complete. Rendering Video..."
echo "========================================="

# 2. Run the Visualizer Script
python visualize.py --grid_size $GRID_SIZE --power_duration $POWER_DURATION

echo "Pipeline executed successfully! Check the 'videos/' directory for the latest timestamped GIF."

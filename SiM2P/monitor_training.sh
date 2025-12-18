#!/bin/bash
# Monitor training progress
# Usage: ./monitor_training.sh [workdir]

WORKDIR="${1:-./SiM2P_workdir/mcsa_mri2pet80_DiT-XL_4_2gpu}"
LOG_FILE="$WORKDIR/log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    echo "Looking for alternatives..."
    LOG_FILE=$(ls -t ./SiM2P_workdir/*/log.txt 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "No log files found!"
        exit 1
    fi
    echo "Using: $LOG_FILE"
fi

echo "========================================"
echo "Monitoring: $LOG_FILE"
echo "========================================"
echo ""

# Show last few training entries
echo "=== Recent Training Progress ==="
tail -100 "$LOG_FILE" | grep -E "^Step|^===|saving|checkpoint" | tail -20

echo ""
echo "=== Checkpoints Saved ==="
ls -la "$WORKDIR"/*.pt 2>/dev/null | tail -10 || echo "No checkpoints yet"

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "=== Training Curve ==="
if [ -f "$WORKDIR/training_curve.png" ]; then
    echo "Training curve available: $WORKDIR/training_curve.png"
    ls -la "$WORKDIR/training_curve.png"
else
    echo "No training curve yet. Run: python plot_training_curve.py --workdir $WORKDIR"
fi

echo ""
echo "========================================"
echo "To follow live: tail -f $LOG_FILE"
echo "========================================"

#!/bin/bash
# Script to launch training in tmux session

SESSION_NAME="yolo_training"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Installing..."
    sudo apt update && sudo apt install -y tmux
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session and run training
echo "🚀 Starting training in tmux session '$SESSION_NAME'..."
tmux new-session -d -s $SESSION_NAME "cd '$PWD' && source .venv/bin/activate && python train.py 2>&1 | tee training.log"

echo ""
echo "✅ Training started successfully!"
echo ""
echo "📋 Useful commands:"
echo "  • View training: tmux attach -t $SESSION_NAME"
echo "  • Detach from session: Ctrl+B then D"
echo "  • Check logs: tail -f training.log"
echo "  • Kill session: tmux kill-session -t $SESSION_NAME"
echo "  • TensorBoard: source .venv/bin/activate && tensorboard --logdir runs/segment/ball_person_model"
echo ""
echo "🔥 Training is running in background. You can close this terminal safely."

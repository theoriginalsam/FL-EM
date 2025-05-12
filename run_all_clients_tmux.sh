#!/bin/bash

# Number of regions/clients
NUM_REGIONS=6

# Server URL
SERVER_URL="http://localhost:5005"

# Create a new tmux session
SESSION_NAME="fl_clients"
tmux new-session -d -s $SESSION_NAME

# Start all clients in separate tmux panes
for i in $(seq 1 $NUM_REGIONS); do
    tmux split-window -v "python run_client.py --region $i --server $SERVER_URL"
    tmux select-layout tiled
done

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME

# python run_client.py --region 1 --server http://localhost:5001
#python run_client.py --region 2 --server http://localhost:5001
# python run_client.py --region 3 --server http://localhost:5001
#python run_client.py --region 4 --server http://localhost:5001
# python run_client.py --region 5 --server http://localhost:5001
#python run_client.py --region 6 --server http://localhost:5001


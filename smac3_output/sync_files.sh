#!/bin/bash

# Variables
REMOTE_USER="nhwpfehl"  # Replace with your username
REMOTE_HOST="transfer.cluster.uni-hannover.de"
REMOTE_DIR="/bigwork/nhwpfehl/architectures-in-rl/smac3_output"
LOCAL_DIR="."  # Replace with your local directory
SSH_KEY="~/.ssh/id_rsa"  # Path to your RSA key

# Rsync command
rsync -avz -e "ssh -i ${SSH_KEY}" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} ${LOCAL_DIR}

# Explanation of options:
# -a: Archive mode (recursive copy and preserves attributes)
# -v: Verbose mode
# -z: Compress file data during the transfer
# -e: Specifies the remote shell to use

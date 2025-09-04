#!/bin/bash
# first serve the vLLM server, then submit the jobs
## serve vLLM server using serve_qwen3.sh and catch the cluster node
sbatch --job-name=qwen3 slurm_serve_qwen3.sh
# wait for the server to start
echo "Waiting for vLLM server to start..."
# get the node where the server is running
SERVER_NODE=$(squeue --me -h -o "%N" -n vllm_server | head -n 1)
# wait until the node is assigned
while [ -z "$SERVER_NODE" ] || [ "$SERVER_NODE" = "(None)" ]; do
    echo "Waiting for node assignment..."
    sleep 60
    SERVER_NODE=$(squeue --me -h -o "%N" -n vllm_server | head -n 1)
done
echo "vLLM server is running on node: $SERVER_NODE" 

method="lmabo-ops"
echo "About to export: server_node='$SERVER_NODE', method='$method'"

# Submit a single job to run all problems in parallel
echo "Submitting single parallel job for all problems"
sbatch --job-name=$method --export=server_node="$SERVER_NODE",method="$method" slurm_run_m.sh
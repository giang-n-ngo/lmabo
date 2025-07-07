#!/bin/bash
# first serve the vLLM server, then submit the jobs
## serve vLLM server using serve_qwen3.sh and catch the cluster node
sbatch --job-name=vllm_server serve_qwen3.sh
# wait for the server to start
echo "Waiting for vLLM server to start..."
sleep 60  # Adjust this sleep time as necessary for your server to start
# get the node where the server is running
SERVER_NODE=$(squeue --me -h -o "%N" -n vllm_server | head -n 1)
# wait until the node is assigned
while [ -z "$SERVER_NODE" ] || [ "$SERVER_NODE" = "(None)" ]; do
    echo "Waiting for node assignment..."
    sleep 5
    SERVER_NODE=$(squeue --me -h -o "%N" -n vllm_server | head -n 1)
done
echo "vLLM server is running on node: $SERVER_NODE" 

# Define the problems and algorithms
## Main problems
# problems=(
    # "Ackley" "Beale" "Branin" "Bukin" "Cosine8" 
    # "DixonPrice" "DropWave" "EggHolder" "Griewank" 
    # "Hartmann" "HolderTable" "Levy" "Michalewicz" "Powell" 
    # "Rastrigin" "Rosenbrock" "Shekel" "SixHumpCamel" "StyblinskiTang" 
    # "Sphere" "BucheRastrigin" "LinearSlope" "AttractiveSector" 
    # "StepEllipsoid" "RosenbrockRotated" "Ellipsoid2" "Discus" "BentCigar" 
    # "SharpRidge" "DifferentPowers" "Weierstrass" "Schaffers" 
    # "SchaffersIllCond" "CompositeGriewankRosenbrock" "Schwefel" "Gallagher21" 
    # "Gallagher101" "Katsuura" "LunacekBiRastrigin" "Easom" 
# )
problems=("Rosenbrock")

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}"
    sbatch --job-name="${problem}" --export=problem=$problem,server_node=$SERVER_NODE run.sh
done
# LMABO: ADAPTIVE ACQUISITION SELECTION FOR BAYESIAN OPTIMIZATION WITH LARGE LANGUAGE MODELS

## Description
A brief description of what this project does and who it's for.

## Setup Environments
```bash
# Create the main environment using environment.yml
conda env create -f environment.yml
# Create a separate environment to host Qwen3 models using vLLM
conda env create -f environment_qwen.yml
```

## Usage
```bash
# To run an experiment with method_name:
python --problem Ackley --method method_name
# To run an experiment with alternating between EI and TS every 3 iterations
python --problem Ackley --method bo_alternating --k 3
```

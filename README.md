# Adaptively Selecting Acquisition Function for Bayesian Optimization Using Large Language Models

Official repository for paper Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models (ICLR 2026).
See paper at: [[Paper link](https://openreview.net/pdf?id=EPKmSgXvRe)]

## Authors
Giang Ngo, Dat Phan Trong, Dang Nguyen, Sunil Gupta, Swetha Venkatesh

## Abstract
Bayesian Optimization critically depends on the choice of acquisition function, but no single strategy is universally optimal; the best choice is non-stationary and problem-dependent. 
Existing adaptive portfolio methods often base their decisions on past function values while ignoring richer information like remaining budget or surrogate model characteristics. 
To address this, we introduce LMABO, a novel framework that casts a pre-trained Large Language Model (LLM) as a zero-shot, online strategist for the BO process. 
At each iteration, LMABO uses a structured state representation to prompt the LLM to select the most suitable acquisition function from a diverse portfolio.
In an evaluation across 50 benchmark problems, LMABO demonstrates a significant performance improvement over strong static, adaptive portfolio, and other LLM-based baselines. 
We show that the LLM's behavior is a comprehensive strategy that adapts to real-time progress, proving its advantage stems from its ability to process and synthesize the complete optimization state into an effective, adaptive policy. 

## TL;DR
LMABO adaptively selects the approprivate acquisition function at each iteration of BO by reading its optimization state, including remaining budget, surrogate model characteristics, and function value range.

## Installation
1. To run the code using Gemini or OpenAI, please provide an API key file as follows:
```python
# key.py
GEMINI_API_KEY = [
    "1st-gemini-api-key-here",
    "2nd-gemini-api-key-here",
]

OPENAI_API_KEY = [
    "1st-openai-api-key-here",
    "2nd-openai-api-key-here",
]
```
2. Install the main environment for LMABO
```
conda env create -f environment.yml
```
For experiments using open-source LLMs, we use vllm to host the LLM. A separate environment for serving with vllm should be installed as follows:
```
conda create -n vllm_server python=3.12 -y
conda activate vllm_server
pip install --upgrade uv
uv pip install vllm --torch-backend=auto
```

## Replicating experiments
1. Main experiments
```
python run.py --problem Ackley --method lmabo # For LMABO
python run.py --problem Ackley --method bo    # For running all static acquisition functions
python run.py --problem Ackley --method gphedge # For GP-Hedge
python run.py --problem Ackley --method esp   # For ESP
python run.py --problem Ackley --method no_past_bo # For No-PASt-BO
python run.py --problem Ackley --method setup_bo # For SETUP-BO
python run.py --problem Ackley --method random_acq # For Random (Full portfolio)
python run.py --problem Ackley --method random_acq_curated1 # For Random (EI, TS, UCB, PosMean)
python run.py --problem Ackley --method random_acq_curated2 # For Random (EI, LogEI, TS)
python run.py --problem Ackley --method bo_alternating --k k # For Alt-EI-TS-k (use specific k)
python run.py --problem Ackley --method bo_explore_exploit # For TwoPhases-TS-EI
```
Add the suffix `_curated` to `gphedge`, `esp`, `no_past_bo`, and `setup_bo` to run these baselines with a curated portfolio of EI, LogEI, and TS.

2. Ablation studies
For structural ablation studies:
```
python run.py --problem Ackley --method lmabo-ab1 # For removing remaining budget
python run.py --problem Ackley --method lmabo-ab2 # For removing GP model characterisitcs
python run.py --problem Ackley --method lmabo-ab3 # For removing shortest distance
python run.py --problem Ackley --method lmabo-ab4 # For removing instruction to avoid ineffective AFs
```
For ablation studies with open-source LLMs, we first need to host open-source LLMs as follows (adjust as needed for your specific hardware setup):
```
conda activate vllm_server
vllm serve Qwen/Qwen3-14B --enable-reasoning --reasoning-parser deepseek_r1 # For LMABO-14B
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --max-model-len 262144 --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.9 --dtype bfloat16 --tensor-parallel-size 4 # For LMABO-30B
vllm serve openai/gpt-oss-120b --dtype auto --gpu-memory-utilization 0.95 # For LMABO-120B
```
We host open-source LLMs at http://NODENAME:8000/v1. Assuming the server node is `NODENAME`, we can run the ablation studies with open-source LLMs as follows:
```
python run.py --problem Ackley --method lmabo-ops --server_node NODENAME # For LMABO-14B
python run.py --problem Ackley --method lmabo-ops3 --server_node NODENAME # For LMABO-30B
python run.py --problem Ackley --method lmabo-ops6 --server_node NODENAME # For LMABO-120B
```
The experiment with GPT-4o mini can be run as follows:
```
python run.py --problem Ackley --method lmabo-gpt
```
The experiment with using context can be run as follows:
```
python run.py --problem Ackley --method lmabo-context
```
3. Statistical significance tests
To replicate the statistical tests mentioned in the paper, run the following command:
```
python report.py
```

## Citation
```bibtex
@inproceedings{ngo2026lmabo,
  title={Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models},
  author={Ngo, Giang and Phan Trong, Dat and Nguyen, Dang and Gupta, Sunil and Venkatesh, Svetha},
  booktitle={Proceedings of the 13th International Conference on Learning Representations},
  year={2026},
  note={Accepted for publication},
  url={https://openreview.net/pdf?id=EPKmSgXvRe}
}
```

# Bridging the Query-Data Semantic Gap: Iterative Query Expansion for Database Retrieval

This work focuses on the task of returning data entries from a table that align with the provided query term when representation misalignment exists between the provided query term and the actual cell values (e.g., synonyms, different spellings, phrasing or concept hierarchy). 

This is the source code of IQER, which improves retrieval performance through iterative query expansion. 

Our paper is submitted to ICDE 2026. 

### Hardware environment
Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz  
NVIDIA A100 80GB  
Note: the experiments do not require the same hardware environment.

## Datasets
We construct three datasets from public data sources to simulate the scenarios with the semantic gap between query and data.
The datasets used in this work are in the folder "dataset"

| Dataset           | Table Size | # Query | Avg. # Answers | Hard Ratio |
|-------------------|------------|---------|----------------|------------|
| Animal            | 6459       | 112     | 6.48           | 18.75      |
| Chemical Compound | 2878       | 74      | 4.05           | 17.57      |
| Product           | 27809      | 100     | 4.27           | 60.00      |

## Run the pipeline
```
python -u main.py --dataset animal --exp_name uct_B100 --budget 100 --k 100 --iterative_check --reform_type multi-aspect --select_query uct
```
You can run the run.py file to run the same setting for the three datasets across three budgets 
```
python -u run.py
```
You need to change the configurations for different LLM servers in .env file.  
In our implementation, we use vLLM to deploy the local LLM server.
```
MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
PORT=1172
API_KEY=qwen
```

## Run the evaluation
```
# dataset exp_name
python evaluation.py animal uct_B100
```
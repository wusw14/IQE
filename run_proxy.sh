# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/animal_qwen.log
# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/chemical_qwen.log
python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm llama --mode proxy > logs/lotus_proxy/animal_llama.log
python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm llama --mode proxy > logs/lotus_proxy/chemical_llama.log
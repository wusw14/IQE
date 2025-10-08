# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/animal_qwen.log
# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/chemical_qwen.log
# python -u eval_lotus.py --dataset animal --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/animal_qwen.log
# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/chemical_qwen.log
# python -u eval_lotus.py --dataset product --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/product_qwen.log
# python -u eval_lotus.py --dataset product --exp_name lotus_proxy --llm qwen --mode proxy > logs/lotus_proxy/product_qwen.log

python -u longcxt.py --dataset chemical_compound --exp_name longcxt --llm qwen > logs/longcxt/chemical_qwen.log
python -u longcxt.py --dataset animal --exp_name longcxt --llm qwen > logs/longcxt/animal_qwen.log
python -u longcxt.py --dataset product --exp_name longcxt --llm qwen > logs/longcxt/product_qwen.log
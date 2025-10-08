# python -u eval_lotus.py --dataset animal --exp_name lotus --llm qwen > logs/lotus/animal_qwen.log
# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus --llm qwen > logs/lotus/chemical_qwen.log
# python -u eval_lotus.py --dataset product --exp_name lotus --llm qwen > logs/lotus/product_qwen.log
# python -u eval_lotus.py --dataset animal --exp_name lotus --llm llama > logs/lotus/animal_llama.log
# python -u eval_lotus.py --dataset chemical_compound --exp_name lotus --llm llama > logs/lotus/chemical_llama.log
# python -u eval_lotus.py --dataset product --exp_name lotus --llm llama > logs/lotus/product_llama.log

python -u longcxt.py --dataset chemical_compound --exp_name longcxt --llm llama > logs/longcxt/chemical_llama.log
python -u longcxt.py --dataset animal --exp_name longcxt --llm llama > logs/longcxt/animal_llama.log
python -u longcxt.py --dataset product --exp_name longcxt --llm llama > logs/longcxt/product_llama.log

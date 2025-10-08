python -u main.py --dataset animal --exp_name llama_wo_select_B100 --budget 100 --k 100 > logs/llama_wo_select/animal_B100.log
python -u main.py --dataset chemical_compound --exp_name llama_wo_select_B100 --budget 100 --k 100 > logs/llama_wo_select/chemical_B100.log
python -u main.py --dataset product --exp_name llama_wo_select_B100 --budget 100 --k 100 > logs/llama_wo_select/product_B100.log

python -u main.py --dataset animal --exp_name llama_wo_select_B200 --budget 200 --k 200 > logs/llama_wo_select/animal_B200.log
python -u main.py --dataset chemical_compound --exp_name llama_wo_select_B200 --budget 200 --k 200 > logs/llama_wo_select/chemical_B200.log
python -u main.py --dataset product --exp_name llama_wo_select_B200 --budget 200 --k 200 > logs/llama_wo_select/product_B200.log
python -u main.py --dataset animal --exp_name llama_wo_select_B500 --budget 500 --k 500 > logs/llama_wo_select/animal_B500.log
python -u main.py --dataset chemical_compound --exp_name llama_wo_select_B500 --budget 500 --k 500 > logs/llama_wo_select/chemical_B500.log
python -u main.py --dataset product --exp_name llama_wo_select_B500 --budget 500 --k 500 > logs/llama_wo_select/product_B500.log

# python -u main.py --dataset animal --exp_name tablerag_wo_select_B100_llama --budget 100 --k 100 > logs/tablerag_wo_select/llama_animal_B100.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_wo_select_B100_llama --budget 100 --k 100 > logs/tablerag_wo_select/llama_chemical_B100.log
# python -u main.py --dataset product --exp_name tablerag_wo_select_B100_llama --budget 100 --k 100 > logs/tablerag_wo_select/llama_product_B100.log
# python -u main.py --dataset animal --exp_name tablerag_wo_select_B200_llama --budget 200 --k 200 > logs/tablerag_wo_select/llama_animal_B200.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_wo_select_B200_llama --budget 200 --k 200 > logs/tablerag_wo_select/llama_chemical_B200.log
# python -u main.py --dataset product --exp_name tablerag_wo_select_B200_llama --budget 200 --k 200 > logs/tablerag_wo_select/llama_product_B200.log
# python -u main.py --dataset animal --exp_name tablerag_wo_select_B500_llama --budget 500 --k 500 > logs/tablerag_wo_select/llama_animal_B500.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_wo_select_B500_llama --budget 500 --k 500 > logs/tablerag_wo_select/llama_chemical_B500.log
# python -u main.py --dataset product --exp_name tablerag_wo_select_B500_llama --budget 500 --k 500 > logs/tablerag_wo_select/llama_product_B500.log
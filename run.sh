exp_name=qwen_wo_select
python -u main.py --dataset animal --exp_name ${exp_name}_B100 --budget 100 --k 100 --iterative_check > logs/${exp_name}/animal_B100.log
python -u main.py --dataset chemical_compound --exp_name ${exp_name}_B100 --budget 100 --k 100 --iterative_check > logs/${exp_name}/chemical_B100.log
python -u main.py --dataset product --exp_name ${exp_name}_B100 --budget 100 --k 100 --iterative_check > logs/${exp_name}/product_B100.log

python -u main.py --dataset animal --exp_name ${exp_name}_B200 --budget 200 --k 200 --iterative_check > logs/${exp_name}/animal_B200.log
python -u main.py --dataset chemical_compound --exp_name ${exp_name}_B200 --budget 200 --k 200 --iterative_check > logs/${exp_name}/chemical_B200.log
python -u main.py --dataset product --exp_name ${exp_name}_B200 --budget 200 --k 200 --iterative_check > logs/${exp_name}/product_B200.log
python -u main.py --dataset animal --exp_name ${exp_name}_B500 --budget 500 --k 500 --iterative_check > logs/${exp_name}/animal_B500.log
python -u main.py --dataset chemical_compound --exp_name ${exp_name}_B500 --budget 500 --k 500 --iterative_check > logs/${exp_name}/chemical_B500.log
python -u main.py --dataset product --exp_name ${exp_name}_B500 --budget 500 --k 500 --iterative_check > logs/${exp_name}/product_B500.log

# python -u main.py --dataset animal --exp_name tablerag_1009_B100_qwen --budget 100 --k 100 --iterative_check > logs/tablerag_1009/qwen_animal_B100.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_1009_B100_qwen --budget 100 --k 100 --iterative_check > logs/tablerag_1009/qwen_chemical_B100.log
# python -u main.py --dataset product --exp_name tablerag_1009_B100_qwen --budget 100 --k 100 --iterative_check > logs/tablerag_1009/qwen_product_B100.log
# python -u main.py --dataset animal --exp_name tablerag_1009_B200_qwen --budget 200 --k 200 --iterative_check > logs/tablerag_1009/qwen_animal_B200.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_1009_B200_qwen --budget 200 --k 200 --iterative_check > logs/tablerag_1009/qwen_chemical_B200.log
# python -u main.py --dataset product --exp_name tablerag_1009_B200_qwen --budget 200 --k 200 --iterative_check > logs/tablerag_1009/qwen_product_B200.log
# python -u main.py --dataset animal --exp_name tablerag_1009_B500_qwen --budget 500 --k 500 --iterative_check > logs/tablerag_1009/qwen_animal_B500.log
# python -u main.py --dataset chemical_compound --exp_name tablerag_1009_B500_qwen --budget 500 --k 500 --iterative_check > logs/tablerag_1009/qwen_chemical_B500.log
# python -u main.py --dataset product --exp_name tablerag_1009_B500_qwen --budget 500 --k 500 --iterative_check > logs/tablerag_1009/qwen_product_B500.log
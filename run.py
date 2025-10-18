import os

exp_name = "early_stop_v2"

iterative_check = True
select_type = "both"
reform_type = "multi-aspect"
early_stop = True

for budget in [100, 200, 500][1:]:
    k = budget
    for dataset in ["animal", "chemical_compound", "product"]:
        cmd = f"python -u main.py --dataset {dataset} --exp_name {exp_name}_B{budget} --budget {budget} --k {k}"
        if iterative_check:
            cmd += " --iterative_check"
        cmd += f" --reform_type {reform_type}"
        cmd += f" --select_query {select_type}"
        if early_stop:
            cmd += f" --early_stop"
        if not os.path.exists(f"logs/{exp_name}"):
            os.makedirs(f"logs/{exp_name}")
        cmd += f" > logs/{exp_name}/{dataset}_B{budget}.log"
        print(cmd)
        os.system(cmd)

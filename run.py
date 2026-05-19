import os
import sys
from dotenv import set_key
import pynvml
import time


def find_available_port():
    if_available = False
    port = None
    while not if_available:
        for i in range(3, 7):
            # if i == 5:
            #     continue
            # check the utilization of device i for every 10s
            if_available = True
            for t in range(6):
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                if utilization.gpu > 10:
                    if_available = False
                    break
                else:
                    time.sleep(5)
            if if_available:
                port = f"112{i}"
                break
        if port is None:
            time.sleep(300)
    print(f"Find port!!!: {port}")
    return port


exp_name = "qwen3.5_rtv"

iterative_check = True
select_type = "uct"
reform_type = "multi-aspect"
early_stop = True
stop_iter = 4

for i in range(1, 6):
    for budget in [100, 200, 500]:
        k = budget
        for dataset in ["animal", "chemical_compound", "product"]:
            port = find_available_port()
            set_key(".env", "PORT", port)
            cmd = f"nohup python -u main.py --dataset {dataset} --exp_name {exp_name}_v{i} --budget {budget} --k {k}"
            if iterative_check:
                cmd += " --iterative_check"
            cmd += f" --reform_type {reform_type}"
            cmd += f" --select_query {select_type}"
            if early_stop:
                cmd += f" --early_stop"
                cmd += f" --stop_iter {stop_iter}"
            if not os.path.exists(f"logs/{exp_name}_v{i}"):
                os.makedirs(f"logs/{exp_name}_v{i}")
            cmd += f" > logs/{exp_name}_v{i}/{dataset}_B{budget}.log 2>&1 &"
            print(cmd)
            os.system(cmd)

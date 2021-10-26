import subprocess
import json
# from ecosys.decorators.eval_decorators import model_eval

def get_gpustat():
    result = subprocess.run(["gpustat", "--json"], stdout=subprocess.PIPE)
    output_json = json.loads(result.stdout)

    return output_json

def measure_gpu_power():
    gpus = get_gpustat()['gpus']
    return sum(gpu['power.draw'] for gpu in gpus)

def get_num_gpus():
    gpus = get_gpustat()['gpus']

    return len(gpus)

if __name__ == "__main__":
    print(measure_gpu_power())
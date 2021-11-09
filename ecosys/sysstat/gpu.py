import subprocess
import json
from datetime import datetime

def list_of_obj_agg(arr, key, opt, idx='all'):
    if type(idx) is str and idx == 'all':
        idx = [x for x in range(len(arr))]
    return opt([a[key] if type(a) is dict else getattr(a, key)for i, a in enumerate(arr) if i in idx])

def get_gpu_stats_raw():
    result = subprocess.run(["gpustat", "--json"], stdout=subprocess.PIPE)
    output_json = json.loads(result.stdout)
    return output_json

def get_gpu_stats_agg(indices='all'):
    gpus_raw = get_gpu_stats_raw()
    gpus = gpus_raw['gpus']
    if type(indices) is str and indices == 'all':
        indices = [x for x in range(get_num_gpus())]
    gpustat = {
        'query_time': datetime.strptime(gpus_raw['query_time'], "%Y-%m-%dT%H:%M:%S.%f").timestamp(),
        'utilization.gpu': list_of_obj_agg(gpus, 'utilization.gpu', sum, indices),
        'power.draw': list_of_obj_agg(gpus, 'power.draw', sum, indices),
        'memory.used': list_of_obj_agg(gpus, 'memory.used', sum, indices),
        'memory.total': list_of_obj_agg(gpus, 'memory.total', sum, indices),
    }
    return gpustat


def measure_gpu_power(indices='all'):
    gpus = get_gpu_stats_raw()['gpus']
    if type(indices) is str and indices == 'all':
        indices = [x for x in range(get_num_gpus())]
    return sum(gpu['power.draw'] for i, gpu in enumerate(gpus) if i in indices)

def get_num_gpus():
    gpus = get_gpu_stats_raw()['gpus']
    return len(gpus)

if __name__ == "__main__":
    print(measure_gpu_power())
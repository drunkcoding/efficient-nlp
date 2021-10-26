import torch
import asyncio
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from network_attached_model import NetworkAttachedModel
from ecosys.sysstat.gpu import measure_gpu_power
from time import sleep
from ecosys.utils.logger import Logger

logger = Logger(__file__, 'INFO', 'a')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lock = threading.Lock()

async def power_generator():
    while lock.locked():
        await asyncio.sleep(0.1)
        yield measure_gpu_power()

async def measure_power(key):
    async for p in power_generator():
        logger.info('%s power %s', key, p)

if __name__ == "__main__":

    base_dir = "/home/oai/share"
    task_name = 'CoLA'
    batch_size = 32

    # lock.acquire()
    # loop = asyncio.new_event_loop()
    # loop.run_until_complete(measure_power('base'))

    # sleep(10)
    # lock.release()

    # sleep(1)
    # loop.close()

    # lock.acquire()
    # loop = asyncio.new_event_loop()
    # loop.run_until_complete(measure_power('serving'))

    model = NetworkAttachedModel(
        AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/distilbert-base-uncased-{task_name}", return_dict=True).to(device)
    )
    model.serve('0.0.0.0:50051')
    

import functools
import torch
from tqdm import tqdm
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

def profile_flops(func):
    @functools.wraps(func)
    def wrapper_profile_flops(*args, **kwargs):
        # print(kwargs, args)
        profiler = FlopsProfiler(kwargs['model'])
        profiler.start_profile()
        with torch.no_grad():
            ret = func(*args, **kwargs)
        profiler.end_profile()
        profiler.print_model_profile()
        return ret
    return wrapper_profile_flops

# def profile_flops(model, dataloader, profile_step = 10):
#     """
#     skip warmup stage of profiling
#     """
#     def profile_flops_decorator(func):
#         @functools.wraps(func)
#         def wrapper_profile_flops(*args, **kwargs):
#             profiler = FlopsProfiler(model)
#             with torch.no_grad():
#                 for idx, batch in enumerate(tqdm(dataloader, desc="Profiling FLOPS")):
#                     input, label = batch
#                     if profile_step == idx:
#                         profiler.start_profile()
#                     logits = model(input)
#                     if profile_step == idx:
#                         profiler.end_profile()
#                         profiler.print_model_profile(profile_step=profile_step)
#                     func(logits, label, *args, **kwargs)
#         return wrapper_profile_flops
#     return profile_flops_decorator
import functools
import torch
from tqdm import tqdm

# def parametrized(dec):
#     def layer(*args, **kwargs):
#         def repl(f):
#             return dec(f, *args, **kwargs)
#         return repl
#     return layer

def model_eval(dataloader):
    def model_eval_decorator(func):
        @functools.wraps(func)
        def wrapper_model_eval(*args, **kwargs):
            with torch.no_grad():
                for input, label in tqdm(dataloader, desc="Evaluating"):
                    func(input, label, *args, **kwargs)
        return wrapper_model_eval
    return model_eval_decorator

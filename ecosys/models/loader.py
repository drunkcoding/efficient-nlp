import torchvision.models as models
import transformers
import torch


def load_tokenizer(tokenizer_type, path, **kwargs):
    if tokenizer_type in dir(transformers):
        tokenizer = getattr(transformers, tokenizer_type)
        return tokenizer.from_pretrained(path, **kwargs)

    raise ValueError(
        "Invalid tokenizer_type: {}, should be one of the model supported by transformers".format(
            tokenizer_type
        )
    )


def load_model(model_type, path=None, do_eval=False, **kwargs):
    """
    Return a Pytorch model instance of `model_type` and loaded form `path` if provided and supported.
    Raise ``ValueError`` if `model_type` not supported.
    """

    if model_type in dir(transformers):
        model_class = getattr(transformers, model_type)
        model = model_class.from_pretrained(path, **kwargs)
    elif model_type in dir(models):
        model_class = getattr(models, model_type)
        model = model_class(**kwargs)
    elif model_type == 'raw':
        model = torch.load(path)
    else:
        raise ValueError(
            "Invalid model_type: {}, should be one of the model supported by transformers or torchvision or raw pytorch".format(
                model_type
            )
        )

    if do_eval:
        model.eval()

    return model

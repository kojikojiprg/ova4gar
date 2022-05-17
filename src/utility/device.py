import torch


def get_device(args):
    if torch.cuda.is_available():
        device = "cuda"
        if args.gpu is not None:
            device += f":{args.gpu}"
    else:
        device = "cpu"

    return device

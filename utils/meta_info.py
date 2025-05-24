import torch

def torch_info():
    print("Torch version: ", torch.__version__)
    print("Torch CUDA version: ", torch.version.cuda)
    print("Is CUDA available: ", torch.cuda.is_available())
    print("GPU: {0} ".format(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU only")
    print("---------------------------------------------------------------\n")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
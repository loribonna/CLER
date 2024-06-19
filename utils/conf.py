import random
import torch
import numpy as np

def get_device() -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device():
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                gpu_memory.append(torch.cuda.memory_allocated(i))
            if all(memory == 0 for memory in gpu_memory):
                print("WARNING: some weird GPU memory issue. Using trick from https://discuss.pytorch.org/t/torch-cuda-memory-allocated-returns-0-if-pytorch-no-cuda-memory-caching-1/188796")
                for i in range(torch.cuda.device_count()):
                    free_memory, total_memory = torch.cuda.mem_get_info(i)
                    gpu_memory[i] = total_memory - free_memory
            device = torch.device(f'cuda:{np.argmin(gpu_memory)}')
            print(f'Using device {device}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("WARNING: MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            print("WARNING: Something went wrong with MPS. Using CPU.")

        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()
        print(f'Using device {get_device.device}')

    return get_device.device


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'

def share_path() -> str:
    """
    Returns the path where to load the shared data.
    """
    return base_path()


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

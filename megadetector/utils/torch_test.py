"""

 torch_test.py

 Simple script to verify CUDA availability, used to verify a CUDA/PyTorch
 environment.

"""

def torch_test():
    """
    Print diagnostic information about Torch/CUDA status, including Torch/CUDA versions
    and all available CUDA device names.
    """
    
    import torch

    print('Torch version: {}'.format(str(torch.__version__)))
    print('CUDA available: {}'.format(torch.cuda.is_available()))

    device_ids = list(range(torch.cuda.device_count()))
    print('Found {} CUDA devices:'.format(len(device_ids)))
    for device_id in device_ids:
        device_name = 'unknown'
        try:
            device_name = torch.cuda.get_device_name(device=device_id)
        except Exception as e:
            pass
        print('{}: {}'.format(device_id,device_name))
       
if __name__ == '__main__':    
    torch_test()

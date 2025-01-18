"""

gpu_test.py

Simple script to verify CUDA availability, used to verify a CUDA environment
for TF or PyTorch

"""

def torch_test():
    """
    Print diagnostic information about Torch/CUDA status, including Torch/CUDA versions
    and all available CUDA device names.
    """
    
    try:
        import torch
    except Exception as e: #noqa
        print('Torch unavailable, not running Torch tests.  Torch import error was:\n{}'.format(
            str(e)))
        return

    print('Torch version: {}'.format(str(torch.__version__)))
    print('CUDA available (according to PyTorch): {}'.format(torch.cuda.is_available()))
    print('CUDA version (according to PyTorch): {}'.format(torch.version.cuda))
    print('CuDNN version (according to PyTorch): {}'.format(torch.backends.cudnn.version()))

    device_ids = list(range(torch.cuda.device_count()))
    print('Found {} CUDA devices:'.format(len(device_ids)))
    for device_id in device_ids:
        device_name = 'unknown'
        try:
            device_name = torch.cuda.get_device_name(device=device_id)
        except Exception as e: #noqa
            pass
        print('{}: {}'.format(device_id,device_name))
        

def tf_test():
    """
    Print diagnostic information about TF/CUDA status.
    """
    
    try:
        import tensorflow as tf
    except Exception as e: #noqa
        print('TF unavailable, not running TF tests.  TF import error was:\nP{}'.format(
            str(e)))
        return
    
    gpus = tf.config.list_physical_devices('GPU')
  
    if gpus:
        print('TensorFlow found the following GPUs:')
        for gpu in gpus:
            print(gpu.name)
            
        from tensorflow.python.platform import build_info as build
        print(f"TF version: {tf.__version__}")
        print(f"CUDA version reported by TF: {build.build_info['cuda_version']}")
        print(f"CuDNN version reported by TF: {build.build_info['cudnn_version']}")
    else:
        print("No GPUs found by TF")
        
      
if __name__ == '__main__':    
    
    print('*** Running Torch tests ***\n')
    torch_test()
    
    print('\n*** Running TF tests ***\n')
    tf_test()

"""

gpu_test.py

Simple script to verify CUDA availability, used to verify a CUDA environment
for TF or PyTorch

"""

# Minimize TF printouts
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

try:
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
except Exception:
    pass


#%% Torch/TF test functions

def torch_test():
    """
    Print diagnostic information about Torch/CUDA status, including Torch/CUDA versions
    and all available CUDA device names.

    Returns:
        int: The number of CUDA devices reported by PyTorch.
    """

    try:
        import torch
    except Exception as e: #noqa
        print('PyTorch unavailable, not running PyTorch tests.  PyTorch import error was:\n{}'.format(
            str(e)))
        return 0

    print('Torch version: {}'.format(str(torch.__version__)))
    print('CUDA available (according to PyTorch): {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('CUDA version (according to PyTorch): {}'.format(torch.version.cuda))
        print('CuDNN version (according to PyTorch): {}'.format(torch.backends.cudnn.version()))

    device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) > 0:
        cuda_str = 'Found {} CUDA devices:'.format(len(device_ids))
        print(cuda_str)

        for device_id in device_ids:
            device_name = 'unknown'
            try:
                device_name = torch.cuda.get_device_name(device=device_id)
            except Exception as e: #noqa
                pass
            print('{}: {}'.format(device_id,device_name))
    else:
        print('No GPUs reported by PyTorch')

    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            print('PyTorch reports that Metal Performance Shaders are available')
    except Exception:
        pass
    return len(device_ids)


def tf_test():
    """
    Print diagnostic information about TF/CUDA status.

    Returns:
        int: The number of CUDA devices reported by TensorFlow.
    """

    try:
        import tensorflow as tf # type: ignore
    except Exception as e: #noqa
        print('TensorFlow unavailable, not running TF tests.  TF import error was:\n{}'.format(
            str(e)))
        return 0

    from tensorflow.python.platform import build_info as build # type: ignore
    print(f"TF version: {tf.__version__}")

    if 'cuda_version' not in build.build_info:
        print('TF does not appear to be built with CUDA')
    else:
        print(f"CUDA build version reported by TensorFlow: {build.build_info['cuda_version']}")
    if 'cudnn_version' not in build.build_info:
        print('TF does not appear to be built with CuDNN')
    else:
        print(f"CuDNN build version reported by TensorFlow: {build.build_info['cudnn_version']}")

    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt # type: ignore
        print("Linked TensorRT version: {}".format(trt.trt_utils._pywrap_py_utils.get_linked_tensorrt_version()))
    except Exception:
        print('Could not probe TensorRT version')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus is None:
        gpus = []

    if len(gpus) > 0:
        print('TensorFlow found the following GPUs:')
        for gpu in gpus:
            print(gpu.name)

    else:
        print('No GPUs reported by TensorFlow')

    return len(gpus)


#%% Command-line driver

if __name__ == '__main__':

    print('*** Running Torch tests ***\n')
    torch_test()

    print('\n*** Running TF tests ***\n')
    tf_test()

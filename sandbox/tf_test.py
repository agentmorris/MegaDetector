########
#
# tf_test.py
#
# Simple script to verify CUDA availability, used to verify a CUDA/TF
# environment.
#
########

import tensorflow as tf

gpu_available = tf.test.is_gpu_available()
print('TF GPU available: {}'.format(gpu_available))

n_gpus = len(tf.config.list_physical_devices('GPU'))
print('TensorFlow indicates that {} GPUs are available'.format(n_gpus))
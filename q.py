import tensorflow as tf

# Check if GPU acceleration is enabled
print("GPU available:" , tf.config.list_physical_devices('GPU'))
print("Is GPU available for TensorFlow:", tf.test.is_gpu_available())
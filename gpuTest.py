import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow is built with GPU support
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Built with GPU support:", tf.test.is_built_with_gpu_support())

# List all GPUs visible to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected by TensorFlow:", gpus)

if gpus:
    try:
        # Set memory growth to avoid TensorFlow using all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialized
        print(e)
else:
    print("No GPUs found or GPU support is not enabled in TensorFlow.")

import tensorflow as tf
from tensorflow.keras.layers import Layer

class ReshapeMxnet2Layer(Layer):
    """
    Custom Keras layer that implements the reshape_mxnet_2 operation.
    
    This layer takes a 4D tensor and performs a complex reshape operation:
    1. Splits the tensor into two halves along the height dimension
    2. For each half, extracts the channels at index 0 and 1
    3. Reorganizes the extracted tensors into a new tensor with 4 channels
    
    Input shape:
        4D tensor with shape (batch_size, height, width, 2)
    
    Output shape:
        4D tensor with shape (batch_size, height/2, width, 4)
    
    Example:
        ```python
        # Input tensor with shape (1, 10, 8, 2)
        x = tf.random.normal((1, 10, 8, 2))
        
        # Apply the layer
        layer = ReshapeMxnet2Layer()
        y = layer(x)
        
        # Output tensor with shape (1, 5, 8, 4)
        ```
    """
    
    def __init__(self, **kwargs):
        super(ReshapeMxnet2Layer, self).__init__(**kwargs)
        
    def call(self, inputs):
        # Calculate half height dynamically using tf.shape
        height = tf.shape(inputs)[1]
        sz = height // 2
        
        # Split tensor into parts
        inter_1 = inputs[:, 0:sz, :, 0]
        inter_2 = inputs[:, 0:sz, :, 1]
        inter_3 = inputs[:, sz:, :, 0]
        inter_4 = inputs[:, sz:, :, 1]
        

        final = tf.stack([inter_1, inter_3, inter_2, inter_4])
        return tf.transpose(final, (1, 2, 3, 0))
    
    def compute_output_shape(self, input_shape):
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, height/2, width, 4)
        # Handle dynamic dimensions (None values)
        batch = input_shape[0]
        height = None if input_shape[1] is None else input_shape[1] // 2
        width = input_shape[2]
        return (batch, height, width, 4)
    
    def get_config(self):
        config = super(ReshapeMxnet2Layer, self).get_config()
        return config

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import concatenate

class ReshapeMxnet1Layer(Layer):
    """
    Custom Keras layer that implements the reshape_mxnet_1 operation.
    
    This layer takes a 4D tensor with shape (batch, height, width, channels)
    where channels must be at least 4, and performs the following operations:
    1. Concatenates the 0th and 1st channels along the height dimension
    2. Concatenates the 2nd and 3rd channels along the height dimension
    3. Stacks these two tensors and transposes the result
    
    Input shape:
        4D tensor with shape (batch_size, height, width, channels)
        where channels >= 4
    
    Output shape:
        4D tensor with shape (batch_size, height*2, width, 2)
    
    Example:
        ```python
        # Input tensor with shape (1, 32, 32, 4)
        x = tf.random.normal((1, 32, 32, 4))
        
        # Apply the layer
        reshape_layer = ReshapeMxnet1Layer()
        y = reshape_layer(x)
        
        # Output tensor with shape (1, 64, 32, 2)
        ```
    """
    
    def __init__(self, **kwargs):
        super(ReshapeMxnet1Layer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Extract first 4 channels (we need exactly 4 channels for this operation)
        channel_0 = inputs[:, :, :, 0]
        channel_1 = inputs[:, :, :, 1]
        channel_2 = inputs[:, :, :, 2]
        channel_3 = inputs[:, :, :, 3]
        
        # Concatenate channels along the height dimension (axis=1)
        inter_1 = concatenate([channel_0, channel_1], axis=1)
        inter_2 = concatenate([channel_2, channel_3], axis=1)
        
        # Stack the intermediate tensors
        # This creates a tensor with shape (2, batch, height*2, width)
        final = tf.stack([inter_1, inter_2])
        
        # Transpose to move the stacked dimension to the end
        # From (2, batch, height*2, width) to (batch, height*2, width, 2)
        transposed = tf.transpose(final, (1, 2, 3, 0))
        
        return transposed
    
    def compute_output_shape(self, input_shape):
        # Input shape: (batch, height, width, channels)
        # Output shape: (batch, height*2, width, 2)
        # Handle dynamic dimensions (None values)
        batch = input_shape[0]
        height = None if input_shape[1] is None else input_shape[1] * 2
        width = input_shape[2]
        return (batch, height, width, 2)
    
    def get_config(self):
        config = super(ReshapeMxnet1Layer, self).get_config()
        return config

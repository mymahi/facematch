import tensorflow as tf
from tensorflow.keras.layers import Layer

class CropLayer(Layer):
    """
    Custom Keras layer that crops an input tensor to match the spatial dimensions 
    of a reference tensor, centering the crop in the spatial dimensions.
    
    This layer takes two inputs:
    1. The tensor to be cropped
    2. A reference tensor whose spatial dimensions (height and width) will be used
       as the target dimensions for cropping the first tensor
    
    Input shape:
        - input[0]: 4D tensor with shape (batch_size, height1, width1, channels1)
        - input[1]: 4D tensor with shape (batch_size, height2, width2, channels2)
          where height2 <= height1 and width2 <= width1
    
    Output shape:
        4D tensor with shape (batch_size, height2, width2, channels1)
    
    Example:
        ```python
        # Input tensor with shape (1, 64, 64, 32)
        x = tf.random.normal((1, 64, 64, 32))
        
        # Reference tensor with shape (1, 32, 32, 16)
        ref = tf.random.normal((1, 32, 32, 16))
        
        # Apply the layer
        crop_layer = CropLayer()
        y = crop_layer([x, ref])
        
        # Output tensor with shape (1, 32, 32, 32)
        ```
    """
    
    def __init__(self, **kwargs):
        super(CropLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # inputs should be a list of two tensors
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("CropLayer expects a list of two tensors as input")
            
        input_tensor = inputs[0]
        obj_shape_tensor = inputs[1]
        
        # Extract dimensions directly using tf.shape with indices
        # Shape components for the input tensor
        batch = tf.shape(input_tensor)[0]
        in_height = tf.shape(input_tensor)[1]
        in_width = tf.shape(input_tensor)[2]
        channels = tf.shape(input_tensor)[3]
        
        # Shape components for the reference tensor
        ref_height = tf.shape(obj_shape_tensor)[1]
        ref_width = tf.shape(obj_shape_tensor)[2]
        
        # Calculate offsets for centering the crop
        height_offset = (in_height - ref_height) // 2
        width_offset = (in_width - ref_width) // 2
        offsets = tf.stack([0, height_offset, width_offset, 0])
        
        # Define the size of the crop using tf.stack
        size = tf.stack([-1, ref_height, ref_width, -1])
        
        # Perform the crop using tf.slice
        cropped = tf.slice(input_tensor, offsets, size)
        
        return cropped
    
    def compute_output_shape(self, input_shape):
        # input_shape is a list of two shapes
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3])
    
    def get_config(self):
        config = super(CropLayer, self).get_config()
        return config

import tensorflow as tf
from crop_layer import CropLayer

def test_crop_layer():
    # Create a simple model using our custom layer
    input1 = tf.keras.layers.Input(shape=(64, 64, 32))
    input2 = tf.keras.layers.Input(shape=(32, 32, 16))
    outputs = CropLayer()([input1, input2])
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
    
    # Print model summary
    model.summary()
    
    # Create sample input tensors
    sample_input1 = tf.random.normal((1, 64, 64, 32))
    sample_input2 = tf.random.normal((1, 32, 32, 16))
    
    # Run the model
    output = model([sample_input1, sample_input2])
    
    # Print shapes for verification
    print(f"Input 1 shape: {sample_input1.shape}")
    print(f"Input 2 shape: {sample_input2.shape}")
    print(f"Output shape: {output.shape}")
    
    # Manually calculate the expected output for verification
    x1_shape = tf.shape(sample_input1)
    x2_shape = tf.shape(sample_input2)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    manual_result = tf.slice(sample_input1, offsets, size)
    
    # Verify the layer works as expected
    is_close = tf.reduce_all(tf.abs(output - manual_result) < 1e-6)
    print(f"Layer implementation matches manual calculation: {is_close}")

if __name__ == "__main__":
    test_crop_layer()

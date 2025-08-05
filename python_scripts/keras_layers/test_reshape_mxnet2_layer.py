import tensorflow as tf
from reshape_mxnet2_layer import ReshapeMxnet2Layer

def test_reshape_mxnet2_layer():
    # Create a simple model using our custom layer
    inputs = tf.keras.layers.Input(shape=(10, 8, 2))
    outputs = ReshapeMxnet2Layer()(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    # Create a sample input tensor
    sample_input = tf.random.normal((1, 10, 8, 2))
    
    # Run the model
    output = model(sample_input)
    
    # Print shapes for verification
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Manually calculate the expected output for verification
    sz = 10 // 2
    inter_1 = sample_input[:, 0:sz, :, 0]
    inter_2 = sample_input[:, 0:sz, :, 1]
    inter_3 = sample_input[:, sz:, :, 0]
    inter_4 = sample_input[:, sz:, :, 1]
    
    # Reshape to add channel dimension
    inter_1 = tf.expand_dims(inter_1, axis=3)
    inter_2 = tf.expand_dims(inter_2, axis=3)
    inter_3 = tf.expand_dims(inter_3, axis=3)
    inter_4 = tf.expand_dims(inter_4, axis=3)
    
    # Combine using concatenate
    top_half = tf.concat([inter_1, inter_3], axis=3)
    bottom_half = tf.concat([inter_2, inter_4], axis=3)
    manual_result = tf.concat([top_half, bottom_half], axis=3)
    
    # Verify the layer works as expected
    is_close = tf.reduce_all(tf.abs(output - manual_result) < 1e-6)
    print(f"Layer implementation matches manual calculation: {is_close}")

if __name__ == "__main__":
    test_reshape_mxnet2_layer()

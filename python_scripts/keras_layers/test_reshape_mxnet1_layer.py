import tensorflow as tf
from reshape_mxnet1_layer import ReshapeMxnet1Layer

def test_reshape_mxnet1_layer():
    # Create a simple model using our custom layer
    inputs = tf.keras.layers.Input(shape=(32, 32, 4))
    outputs = ReshapeMxnet1Layer()(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    # Create a sample input tensor
    sample_input = tf.random.normal((1, 32, 32, 4))
    
    # Run the model
    output = model(sample_input)
    
    # Print shapes for verification
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Manually calculate the expected output for verification
    channel_0 = sample_input[:, :, :, 0]
    channel_1 = sample_input[:, :, :, 1]
    channel_2 = sample_input[:, :, :, 2]
    channel_3 = sample_input[:, :, :, 3]
    
    inter_1 = tf.concat([channel_0, channel_1], axis=1)
    inter_2 = tf.concat([channel_2, channel_3], axis=1)
    
    inter_1 = tf.expand_dims(inter_1, axis=-1)
    inter_2 = tf.expand_dims(inter_2, axis=-1)
    
    manual_result = tf.concat([inter_1, inter_2], axis=-1)
    
    # Verify the layer works as expected
    is_close = tf.reduce_all(tf.abs(output - manual_result) < 1e-6)
    print(f"Layer implementation matches manual calculation: {is_close}")

if __name__ == "__main__":
    test_reshape_mxnet1_layer()

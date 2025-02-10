import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import onnx
import numpy as np

# Define the Functional API model
input_layer = Input(shape=(2,))  # Replace 10 with your input dimension
output_layer = Dense(1, activation='sigmoid')(input_layer)
functional_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model (optional, for training purposes)
functional_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
functional_model.summary()

# Convert to ONNX
import tf2onnx
input_signature = [tf.TensorSpec([None, 2], tf.float32, name="input")]  # Adjust input shape accordingly
onnx_model, _ = tf2onnx.convert.from_keras(functional_model, input_signature, opset=13)

onnx.save(onnx_model, "sample_sigmoid.onnx")


# inputs = [np.array([[-100, -100]]), np.array([[100, -100]]), np.array([[-100, -100]]), np.array([[-100, -100]])]
# for input in inputs:
#     output = functional_model.predict(input)
#     print(input, output)


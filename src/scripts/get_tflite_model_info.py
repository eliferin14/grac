import tensorflow as tf
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("--model_path", type=str, help="Path to the model file")
args = parser.parse_args()

# Load the model
model = tf.lite.Interpreter(args.model_path)
model_input = model.get_input_details()
model_output = model.get_output_details()
model.allocate_tensors()

print(f"\nModel input: {model_input}")
print(f"\nModel output: {model_output}")

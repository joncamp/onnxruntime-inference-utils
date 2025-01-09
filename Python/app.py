import argparse
import onnxruntime as ort
import numpy as np
import time

def load_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def get_input_shape(session):
    inputs = session.get_inputs()
    if len(inputs) == 0:
        print("Model has no inputs defined.")
        exit(1)
    input_shapes = {input.name: input.shape for input in inputs}
    return input_shapes

def generate_random_inputs(input_shapes):
    random_inputs = {}
    for name, shape in input_shapes.items():
        # Replace None or dynamic dimensions with 1 for simplicity
        shape = [dim if isinstance(dim, int) else 1 for dim in shape]
        random_inputs[name] = np.random.random(shape).astype(np.float32)
    return random_inputs

def run_inference(session, random_inputs, iterations):
    try:
        start_time = time.time()
        for _ in range(iterations):
            outputs = session.run(None, random_inputs)
        end_time = time.time()
        print(f"Inference completed {iterations} times in {end_time - start_time:.4f} seconds.")
        return outputs
    except Exception as e:
        print(f"Error during inference: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ONNX model inference from the command line.")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model.")
    parser.add_argument("iterations", type=int, help="Number of inference iterations to run.")
    
    args = parser.parse_args()
    
    # Load the model
    session = load_model(args.model_path)

    # Get input tensor details
    input_shapes = get_input_shape(session)
    print(f"Input tensors and shapes: {input_shapes}")

    # Generate random inputs
    random_inputs = generate_random_inputs(input_shapes)

    # Run inference
    outputs = run_inference(session, random_inputs, args.iterations)

    print("Sample output from the last inference:")
    print(outputs)

if __name__ == "__main__":
    main()

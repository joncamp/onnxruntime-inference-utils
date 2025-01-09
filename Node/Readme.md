# Onnxruntime Interence with Node.js

# Setup

# Usage

`node app.js --model="<model-path>" --iterations=<iterations> --inputs="<input-name:dim1,dim2,...;...>"`

# Examples

## Using the simple model from the Onnxruntime library
`node index.js --model="..\Models\model.onnx" --iterations=10 --inputs="a:3,4;b:4,3"`

``` powershell
Loading model from: ..\Models\model.onnx
User-specified input information:
  Input 0: Name = a, Shape = [3, 4]
  Input 1: Name = b, Shape = [4, 3]
Running inference...
Inference completed over 10 iterations.
Average time per iteration: 13.83 ms
```

## Using an Onnx mediapipe face mesh model 
`node .\index.js --model ..\Models\face_mesh.onnx --iterations 1000 --inputs "input_1:1,192,192,3"`

``` powershell
Loading model from: ..\Models\face_mesh.onnx
User-specified input information:
  Input 0: Name = input_1, Shape = [1, 192, 192, 3]
Running inference...
Inference completed over 1000 iterations.
Average time per iteration: 28.43 ms
```

## QNN Testing

The `useQnn` argument will use the QNN (Qualcomm NPU), but only if an appropriate onnxruntime-node is provided and has support. Generally this is accomplished by deleting `node_modules\onnxruntime-node` and replacing it with the appropriate version.

## Determining the input sizes

Per https://github.com/microsoft/onnxruntime/discussions/17682, the input sized cannot be determined via the javascript implementation of Onnxruntime. To determine the input sizes, you can use the [Python](..\Python) script.

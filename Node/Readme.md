# Onnxruntime Interence with Node.js

`node app.js --model="<model-path>" --iterations=<iterations> --inputs="<input-name:dim1,dim2,...;...>"`

# Determining the input sizes

Per https://github.com/microsoft/onnxruntime/discussions/17682, the input sized cannot be determined via the javascript implementation of Onnxruntime. To determine the input sizes, you can use the python script.

// Import required modules
const ort = require('onnxruntime-node');
const { performance } = require('perf_hooks');
const yargs = require('yargs');

// Function to generate random input tensor
function generateRandomTensor(dims) {
  const size = dims.reduce((a, b) => a * b, 1);
  const data = Float32Array.from({ length: size }, () => Math.random());
  return new ort.Tensor('float32', data, dims);
}

// Parse dimensions from string
function parseDimensions(dimStr) {
  return dimStr.split(',').map(Number);
}

// Main function
async function main() {
  const argv = yargs
    .option('model', {
      alias: 'm',
      type: 'string',
      description: 'Path to the ONNX model file',
      demandOption: true
    })
    .option('iterations', {
      alias: 'i',
      type: 'number',
      description: 'Number of iterations to run',
      demandOption: true
    })
    .option('inputs', {
      alias: 'in',
      type: 'string',
      description: 'Input names and dimensions in the format <input-name:dim1,dim2,...;...>',
      demandOption: true
    })
    .option('useQnn', {
      alias: 'qnn',
      type: 'boolean',
      description: 'Use QNN operator set for quantized models'
    })
    .help()
    .argv;

  const modelPath = argv.model;
  const numIterations = argv.iterations;
  const inputsArg = argv.inputs;

  if (isNaN(numIterations) || numIterations <= 0) {
    console.error('Iterations must be a positive integer.');
    process.exit(1);
  }

  // Parse input names and dimensions
  const inputs = inputsArg.split(';').map(arg => {
    const [name, dims] = arg.split(':');
    if (!name || !dims) {
      console.error(`Invalid input specification: ${arg}`);
      process.exit(1);
    }
    return { name, shape: parseDimensions(dims) };
  });

  try {
    console.log(`Loading model from: ${modelPath}`);

    let inferenceSessionOptions = {}
    if (argv.useQnn) {
      console.log('Using QNN operator set for quantized models');
      inferenceSessionOptions = {
            logSeverityLevel: 0,
            executionProviders: [{ 
                name: 'qnn',
                // backend_path: 'QnnHtp.dll' 
            }]
        }
    }

    const session = await ort.InferenceSession.create(modelPath, inferenceSessionOptions);

    console.log('User-specified input information:');
    inputs.forEach((input, idx) => {
      console.log(`  Input ${idx}: Name = ${input.name}, Shape = [${input.shape.join(', ')}]`);
    });

    // Generate random inputs
    const feeds = {};
    inputs.forEach(input => {
      feeds[input.name] = generateRandomTensor(input.shape);
    });

    console.log('Running inference...');
    const timings = [];

    for (let i = 0; i < numIterations; i++) {
      const start = performance.now();
      await session.run(feeds);
      const end = performance.now();
      timings.push(end - start);
    }

    // Calculate and display timing statistics
    const total = timings.reduce((sum, t) => sum + t, 0);
    const avgTime = total / timings.length;

    console.log(`Inference completed over ${numIterations} iterations.`);
    console.log(`Average time per iteration: ${avgTime.toFixed(2)} ms`);
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

main();

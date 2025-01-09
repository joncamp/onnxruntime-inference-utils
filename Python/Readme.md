# Setup

It is recommended you use a virtual environment.

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

# Usage

`usage: app.py [-h] model_path iterations`

# Example

`python .\app.py ..\Models\model.onnx 10`

``` powershell
Input tensors and shapes: {'a': [3, 4], 'b': [4, 3]}
Inference completed 10 times in 0.0009 seconds.
Sample output from the last inference:
[array([[0.28378332, 0.2823652 , 0.2553254 ],
       [1.040132  , 0.7348061 , 1.1577816 ],
       [0.6118688 , 0.8980346 , 0.5556448 ]], dtype=float32)]
```

`python .\app.py ..\Models\face_mesh.onnx 10`

``` powershell
Input tensors and shapes: {'input_1': [1, 192, 192, 3]}
Inference completed 10 times in 0.0244 seconds.
Sample output from the last inference:
[array([[[[ 96.94706 , 117.668396, -14.586563, ..., 139.89197 ,
           72.10356 ,  14.086828]]]], shape=(1, 1, 1, 1404), dtype=float32), array([[[[-10.579647]]]], dtype=float32)]
```


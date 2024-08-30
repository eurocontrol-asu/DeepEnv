# DeepEnv <img src="https://github.com/JarryGabriel/DeepEnv/blob/main/logo.png" width="100">

This repository contains the DeepEnv library to use deep learning models for aircraft environmental impact assessment.

## Easy Install

For a trouble-free installation, creating a dedicated anaconda environment is recommended :

```sh
conda create -n deepenv python=3.9 -c conda-forge
```

Activate the conda environment :

```sh
conda activate deepenv
```

Install this library:

```sh
git clone TODO
cd deepenv
pip install .

```

## Example of use

Here is a minimal working example:

```python
import pandas as pd
from DeepEnv import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame({
  "typecode": ["A320-214", "A320-214", "A320-214", "A320-214"],
  "groundspeed": [400, 410, 420, 430],
  "altitude": [10000, 11000, 12000, 13000],
  "vertical_rate": [2000, 1500, 1000, 500],
  "mass": [60000, 60000, 60000, 60000],
  
  # optional features:
  "second": [0.0, 1.0, 2.0, 3.0],
  "airspeed": [400, 410, 420, 430],
  
})

flight_fuel = fe.estimate(flight)  # flight.data if traffic flight
```

Note:

- When the `second` column is provided, the fuel estimation is more accurate,
  especially due to **derivatives of speeds** (acceleration) used in the estimation.
- `airspeed` is optional. If not provided, it is assumed to be equal
  to groundspeed. However, accurate airspeed is recommended for better estimation.
- Expected sampling rate is 4 seconds, higher or lower sampling rate might induce noisier fuel flow. Resampling data before estimating fuel flow is recommanded.

For a more complete example, refer to `examples/fuel_estimation.ipynb`

## Aircraft data and estimation models

Aircraft parameters from open data to feed the model are available in `data/aircraft_params.csv` and loaded by default. Model data is available in `models/` and also loaded by default.

You can specify your own data and model file with the following initialization of `FuelEstimator`. You need to make sure the same column names are in your aircraft CSV file.

```python
fe = FuelEstimator(
    aircraft_params_path="path/to/your/data.csv",
    model_path="path/to/your/SavedModel/",
)
```

## Avalaible Models

- DeepBADA4.2.1

## Credits

```bibtex
@inproceedings{jarry2024towards,
  title={On the Generalization Properties of Deep Learning
for Aircraft Fuel Flow Estimation Models},
  author={Jarry, Gabriel and Very, Philippea and Dalmau, Ramon and Sun, Junzi},
  booktitle={SESAR Innovation Days},
  year={2024}
```


}

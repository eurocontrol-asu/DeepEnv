<!--  <img src="https://github.com/JarryGabriel/DeepEnv/blob/main/logo.png" width="250">  -->




# DeepEnv

This repository hosts the DeepEnv library, a collection of research initiatives from the EUROCONTROL Aviation Sustainability Unit and partners. DeepEnv is designed to facilitate the use, construction and experimentation of deep learning models for assessing the environmental impact of aviation. DeepEnv models and codes are made available under a EUPL-1.2 license.

&nbsp;

<ins> **Please note that this repository does not constitute a regulatory framework and should only be used for research purposes, not for operational applications.** </ins> EUROCONTROL disclaims any responsibility for the misuse of these models. 



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
git clone https://github.com/eurocontrol-asu/DeepEnv.git
cd deepenv
pip install .
```

## Avalaible Models

### Single Engine taxiing

- SET_A320_V0.1 : Single engine taxiing classification and localization

  ###### Example of use
  
  For an example of use, refer to `examples/set_estimator/example.ipynb`
  
  Note:
  
  - When the `second` column is provided, the set estimator is more accurate,
    especially due to **derivatives of speeds and track angle** used in the model.
    - Expected sampling rate is 1 seconds, higher or lower sampling rate might induce errors. Resampling data before applying the set estimator is recommanded.

  #### Model reference 
  
  This model is the implementation of the following paper: 
  ```bibtex
  @inproceedings{jarry2024taxiing,
    title={On the Detection of Aircraft Single Engine Taxi using Deep Learning Models},
    author={Jarry, Gabriel and Very, Philippe and Dalmau, Ramon and Delahaye, Daniel and Houndant, Arhtur},
    booktitle={Submitted to SESAR Innovation Days 2024},
    year={2024}
  }
  ```

## In coming Models

### Fuel models :


- DeepBada4.2.1_FUEL_FLOW_V0.1 : BADA 4.2.1 Fuel flow surrogate model (need a Bada Licence)
  ###### Example of use
  
  ```python
  import pandas as pd
  from DeepEnv import FuelEstimator
  
  fe = FuelEstimator(
      aircraft_params_path="PATH_TO/aircraft_params.csv",
      model_path="PATH_TO/DeepBada4.2.1_FUEL_FLOW_V0.1"
  )
  
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
  
  For a more complete example, refer to `examples/fuel_estimator/example.ipynb`

  #### Model reference 
  
  This model is the implementation of the following paper: 
  ```bibtex
  @inproceedings{jarry2024generalization,
    title={On the Generalization Properties of Deep Learning
  for Aircraft Fuel Flow Estimation Models},
    author={Jarry, Gabriel and Very, Philippe and Dalmau, Ramon and Sun, Junzi},
    booktitle={Submitted to SESAR Innovation Days 2024},
    year={2024}
  }
  ```


## Avalaible Training Process

### DeepContrails

  This module aim at building deep learning models to detect contrails on remote sensors (satellite, cameras...)

  **WORK IN PROGRESS** this module is currently standalone.

  This code was use in the following paper: 
  ```bibtex
  @inproceedings{jarry2024segmentation,
    title={Deep Semantic Contrails Segmentation of GOES-16 Satellite Images: An Hyperparameter Exploration},
    author={Jarry, Gabriel and Torjman--Levavasseur, Valentin and Very, Philippe and Heffar, Amine},
    booktitle={Submitted to SESAR Innovation Days 2024},
    year={2024}
  }
  ```


## Credits

To cite this python library use:

  ```bibtex
  @misc{jarry2024deepenv,
    title={DeepEnv: Python library for aircraft environmental impact assessment using Deep Learning},
    author={Jarry, Gabriel and Very, Philippe and Dalmau, Ramon and Sun, Junzi},
    year={2024},
    note={\url{https://doi.org/10.5281/zenodo.13754838}, \url{https://github.com/eurocontrol-asu/DeepEnv}}
  }
  ```







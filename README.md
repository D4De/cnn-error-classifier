# NVDLA Error Classifier
This repository contains the classification tools used to analyze NVDLA's results. It produces accurate reports of spatial patterns identified and domain distribution. It can also create visualizations of the identified errors and automatically generate the JSON files used by CLASSES to perform error simulations.

# Table of contents

1. [Copyright & License](#copyright--license)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Folder Structure](#folder-structure)
    2. [Running the tools](#running-the-tool)
    3. [Options](#options)

## Copyright & License

Copyright (C) 2023 Politecnico di Milano.

This framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Neither the name of Politecnico di Milano nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# Dependencies 
The following libraries are required for this software to run correctly. 

* contourpy
* cycler
* fonttools
* kiwisolver
* matplotlib
* numpy
* packaging
* Pillow
* pyparsing
* python-dateutil
* six
* tqdm

We provide a `requirements.txt` file that can be easily used to install all the necessary libraries, as explained in the [Installation](#installation) section.

# Installation 
We suggest creating a virtual environment either using [Conda](https://docs.conda.io/en/latest/) or [Venv](https://docs.python.org/3/library/venv.html).
To make one using the provided `requirements.txt`, execute the following command
```
conda create --name <env> --file requirements.txt
```
replacing `<env>` with the name of the environment. Then you can activate the environment by running 
```
conda activate <env>
```

# Usage
To correctly use the tool, we must first provide the corrupted tensors in a structure compatible with the classifier.

## Folder structure
```
error-classifier/
    ├── src/
    │   ├── main_nvdla.py
    │   └── ...
    ├── README.md
    ├── requirements.txt
    └── results_operator1/
        ├── ctrl/
        │   ├── golden.npy
        │   ├── hw_unit1/
        │   |   ├── errors.csv
        │   |   └── errors.npz
        │   ├── ...
        │   └── hw_unitN
        └── data/
            ├── golden.npy
            ├── hw_unit1/
            |   ├── errors.csv
            |   └── errors.npz
            ├── ...
            └── hw_unitN
```

## Running the tool
If the injection results follow the supported structure, we can execute the tool and classify the tensors. To do so, we need to run the following command

```bash
python src/main_nvdla.py <operator_folder> <golden_tensor_location> <errors_location> <output_folder> [options]
```

As an example, suppose to have operator folder `conv1` structured as explained above. In that case, the command would be

```bash
python src/main_nvdla.py </path/to/conv1> ../golden.npy errors.npz </path/to/output/folder> [options]
```

### Options
The following options are the most relevant ones for an NVDLA injection campaign. Run the command with the `--help` flag to see a complete list.

* #### **Visualization**
    This tool is capable of creating visualizations of the identified errors. Adding the flag `-v` or `--visualize` will enable this functionality. The images produced will be grouped together according to the underlying spatial pattern. 
    N.B. creating such a visualization is costly and will make the execution of the tool slower.
* #### **Parallelism**
    To speed up the execution of the tool, it is possible to enable multiprocessing. To do so, use the flag `-p N`, which will spawn `N` threads working in parallel. 
* #### **CLASSES Models**
    The goal of performing fault injections is to create error models that CLASSES can use. This tool can make the required JSON files during the analysis to aid this process. To enable this process, use the flag `--classes Sx Operator`, where Sx is the number of the experiment, and Operator is the name of the currently analyzed operator. I.e., if you are creating the 4th model for the convolution, use the flag `--classes S4 Conv`. NOTE: This naming convention is deprecated, but two string arguments after `--classes` are still required. You do not have to follow striclty the naming convention. If the flag is `--classes A B` the file will be named `A_B.json`.
* #### **Error models for each hardware unit**
    If you wish to study the behavior of a single hardware unit, use option `--classes-unit-models`. This will generate one error model
    per hardware unit.
    N.B. This requires the `--classes` option to be enabled, otherwise model generation will fail.
* #### **Epsilon**
    By default, this classifier considers an error in each value that differs from the golden version by a value greater than 1e-3. Using the flag `-eps VAL`, we can specify a different threshold for the classifier. 
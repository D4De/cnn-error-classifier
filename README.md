# NVBitFI Error Classifier
This repository contains the classification tools used to analyze NVBitFI's results. It produces accurate reports of spatial patterns identified and domains distribution. It is also capable of creating visualizations of the identified errors and automatically generate the jsons used by CLASSES to perform error simulation. 

# Table of contents

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

We provide a `requirements.txt` file that can be easily used to install all the necessary libraries as explained in the [Installation](#installation) section.

# Installation 
We suggest creating a virtual environment either using [Conda](https://docs.conda.io/en/latest/) or [Venv](https://docs.python.org/3/library/venv.html).
To create one using the provided `requirements.txt` simply execute the following command
```
conda create --name <env> --file requirements.txt
```
replacing `<env>` with the name of the environment. Then you can activate the environment by running 
```
conda activate <env>
```

# Usage
To correctly use the tool we first need to provide the corrupted tensors in a structure that is compatible with the classifier. 
## Folder structure
```
error-classifier/
    ├── src/
    │   ├── main.py
    │   └── ...
    ├── README.md
    ├── requirements.txt
    └── results_operator1/
        ├── batch1/
        │   └── test/
        │       ├── golden.npy
        │       └── injection_mode/
        │           ├── error1.npy
        │           ├── error2.npy
        │           ├── ...
        │           └── errorN.npy
        ├── batch2
        ├── ...
        └── batchN
```
At the top level of the repository `cnn-error-classifier` we have a `src` folder that contains all the files needed by the tool. The `requirements.txt` file and some other files that can be ignored in this section. 

We need to create a new folder for each operator that we are targeting with the injections. Inside this directory, called `results_operator1` in the above example, we will create one folder for each batch of tests that we executed giving it the following structure.
```
batchX/
    └── test/
        ├── golden.npy
        └── injection_mode1/
            ├── error1.npy
            ├── error2.npy
            ├── ...
            └── errorN.npy
        ├── ...
        └── injection_modeN/
            ├── error1.npy
            ├── error2.npy
            ├── ...
            └── errorN.npy
        
```
Each batch should have a subfolder called `test` inside which we find the following 
* `golden.npy` the NumPy array of the expected result that will be used for reference against each corrupted tensor of the batch
* `injection_mode` one folder for each injection mode adopted that contains all the corrupted tensors produced by NBBitFI.

## Running the tool
If the results of the injection follow the supported structure we can execute the tool and classify the tensors. To do so we need to run the following command
```bash
python src/main.py <operator_folder> <golden_tensor_location> test <output_folder> <options>
```
where the arguments are the following 
* `<operator_folder>` is the name of the folder that contains all the results of a given operator. In the example above it is `results_operator1`.
* `<golden_tensor_location>` is the location of the golden tensor with respect to each batch folder. In the example above it is `test/golden.npy`. 
* `<output_folder>` is the path to the output folder where the results of the analysis will be stored. It is not necessary that this folder exists, the tool will automatically check and create it if needed. 

This program also supports a set of options that can be enabled with apposite flags. 

### Options
The following options can be activated through specific flags
* #### **Data format**
    The default data format adopted by the classifier is  NCHW. It is possible to analyze tensors in the NHWC format by appending the flag `-nhwc` 
* #### **Visualization**
    This tool is capable of creating visualizations of the errors identified. Adding the flag `-v` or `--visualize` will enable this functionality. The images produced will be organized based on the spatial pattern. 
    N.B. Creating such visualization is a costly operation and will make the execution of the tool slower.
* #### **Parallelism**
    To speedup the execution of the tool it is possible to enable multiprocessing. To do so use the flag `-p N` which will spawn `N` threads working in parallel. 
* #### **CLASSES Models**
    The objective of the fault injections is to create error models that can be used by CLASSES. To aid in this process this tool is capable of creating the required json files during the analysis. To enable this process use the flag `--classses Sx Operator`, where Sx is the number of the experiment and Operator is the name of the currently analyzed operator. I.e., if you are  creating the 4th model for the convolution use the flag `--classes S4 Conv`.
* #### **Epsilon**
    By default this classifier considers an error each value that differs from the golden version by a value greater than 1e-3. Using the flag `-eps VAL` we can specify a different threshold for the classifier. 
* #### **Partial reports**
    Using the flag `-pr` the tool will create partial reports for each batch of a given operator. 
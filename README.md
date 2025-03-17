# NVBitFI Error Classifier
This repository contains the classification tools used to analyze NVBitFI's results. It produces accurate reports of spatial patterns identified and domain distribution. It can also create visualizations of the identified errors and automatically generate the JSON files used by CLASSES to perform error simulations.

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
To correctly use the tool, we must first provide the corrupted tensors in a compatible structure with the classifier.

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
At the top level of the repository `cnn-error-classifier`, we have a `src` folder containing all the files the tool needs.
We must create a new folder for each operator we target with the injections. Inside this directory, called `results_operator1` in the above example, we will create one folder for each batch of tests we executed, giving it the following structure.
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
Each batch should have a subfolder called `test` inside which we find the following.
* `golden.npy` the NumPy array of the expected result that will be used for reference against each corrupted tensor of the batch
* `injection_mode` one folder for each injection mode adopted that contains all the corrupted tensors produced by NBBitFI.

## Running the tool
If the injection results follow the supported structure, we can execute the tool and classify the tensors. To do so, we need to run the following command

```bash
python src/main.py <operator_folder> <golden_tensor_location> test <output_folder> <options>
```

where the arguments are the following
* `<operator_folder>`is the folder name that contains all the results of a given operator. In the example above, it is `results_operator1`.
* `<golden_tensor_location>` is the location of the golden tensor with respect to each batch folder. In the example above, it is `test/golden.npy`. 
* `<output_folder>` is the path to the output folder to store the analysis results. This folder doesn't need to exist. The tool will automatically check and create it if needed.
This program also supports options that can be enabled with suitable flags.

### Example run

Download an example input for the classifier from [here](https://miele.faculty.polimi.it/batch_conv_3_with_igprofile.tar.gz).

Unzip it using the command :
```
tar xzvf batch_conv_3_with_igprofile.tar.gz 
```

Then execute:
```
cd src
```

```
python main.py ../tests_2023-04-16_11-00-25 test/output_1.npy test ../output_test --classes conv gemm
```


This command:
Executes the classifier reading from the extracted test folder with the nvbitfi resuts.

It reads relatively from each test folder (conv_1, conv_2, ...):
* the golden output: ``test/output_1.npy``  
* the folder where corrupted output subfolders (fp32_wrv, gp_wrv) are located: ``test``

And outputs in the test folder generating also the classes models. 

### Options
The following options can be activated through specific flags
* #### **Data format**
    The default data format adopted by the classifier is  NCHW. Analyzing tensors in the NHWC format is possible by appending the flag `-nhwc`. 
* #### **Visualization**
    This tool is capable of creating visualizations of the errors identified. Adding the flag `-v` or `--visualize` will enable this functionality. The images produced will be organized based on the spatial pattern. 
    N.B. Creating such a visualization is costly and will make the execution of the tool slower.
* #### **Parallelism**
    To speed up the execution of the tool, it is possible to enable multiprocessing. To do so, use the flag `-p N`, which will spawn `N` threads working in parallel. 
* #### **CLASSES Models**
    The goal of performing fault injections is to create error models that CLASSES can use. This tool can make the required JSON files during the analysis to aid this process. To enable this process, use the flag `--classes Sx Operator`, where Sx is the number of the experiment, and Operator is the name of the currently analyzed operator. I.e., if you are creating the 4th model for the convolution, use the flag `--classes S4 Conv`. NOTE: This naming convention is deprecated, but two string arguments after `--classes` are still required. You do not have to follow striclty the naming convention. If the flag is `--classes A B` the file will be named `A_B.json`.
* #### **Epsilon**
    By default, this classifier considers an error in each value that differs from the golden version by a value greater than 1e-3. Using the flag `-eps VAL`, we can specify a different threshold for the classifier. 
* #### **Partial reports**
    Using the flag `-pr`, the tool will create partial reports for each operator's batch. 

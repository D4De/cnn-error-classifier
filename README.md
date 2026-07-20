# Error Classifier
This repository contains the classification tools used to analyze the fault injection results of NVBitFI and NVDLA. It produces accurate reports of spatial patterns identified and domain distribution. It can also create visualizations of the identified errors and automatically generate the JSON files used by CLASSES to perform error simulations.

The NVDLA version of the classifier processes results of fault injection campaigns executed by using the RTL-level simulator previously adopted in [this work](https://ieeexplore.ieee.org/document/10568018).

# Table of contents

1. [Copyright & License](#copyright--license)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage with NVBitFI](#usage-with-nvbitfi)
    1. [Folder Structure](#folder-structure-for-nvbitfi-results)
    2. [Running the tools](#running-the-tool-(nvbitfi))
    3. [Options](#options)
    4. [Example Run](#example-run)
5. [Usage with NVDLA](#usage-with-nvdla)
    1. [Folder Structure](#folder-structure-for-nvdla-results)
    2. [Running the tool](#running-the-tool-(nvdla))
    3. [Options](#options-(nvdla))
    4. [Channel counting](#channel-counting-(nvdla))
    5. [Extending the set of recognized spatial classes](#extending-the-set-of-recognized-spatial-classes)
    6. [NVDLA example run](#nvdla-example-run)

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

# Usage with NVBitFI
To correctly use the tool, we must first provide the corrupted tensors in a structure compatible with the classifier.

## Folder structure for NVBitFI results
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

## Running the tool (NVBitFI)
If the injection results follow the supported structure, we can execute the tool and classify the tensors. To do so, we need to run the following command

```bash
python src/main.py <operator_folder> <golden_tensor_location> test <output_folder> <options>
```

where the arguments are the following
* `<operator_folder>`is the folder name that contains all the results of a given operator. In the example above, it is `results_operator1`.
* `<golden_tensor_location>` is the location of the golden tensor with respect to each batch folder. In the example above, it is `test/golden.npy`. 
* `<output_folder>` is the path to the output folder to store the analysis results. This folder doesn't need to exist. The tool will automatically check and create it if needed.
This program also supports options that can be enabled with suitable flags.

## Options
The following options can be activated through specific flags
* ### **Data format**
    The default data format adopted by the classifier is  NCHW. Analyzing tensors in the NHWC format is possible by appending the flag `-nhwc`. 
* ### **Visualization**
    This tool is capable of creating visualizations of the errors identified. Adding the flag `-v` or `--visualize` will enable this functionality. The images produced will be organized based on the spatial pattern. 
    N.B. Creating such a visualization is costly and will make the execution of the tool slower.
* ### **Parallelism**
    To speed up the execution of the tool, it is possible to enable multiprocessing. To do so, use the flag `-p N`, which will spawn `N` threads working in parallel. 
* ### **CLASSES Models**
    The goal of performing fault injections is to create error models that CLASSES can use. This tool can make the required JSON files during the analysis to aid this process. To enable this process, use the flag `--classes Sx Operator`, where Sx is the number of the experiment, and Operator is the name of the currently analyzed operator. I.e., if you are creating the 4th model for the convolution, use the flag `--classes S4 Conv`. NOTE: This naming convention is deprecated, but two string arguments after `--classes` are still required. You do not have to follow striclty the naming convention. If the flag is `--classes A B` the file will be named `A_B.json`.
* ### **Epsilon**
    By default, this classifier considers an error in each value that differs from the golden version by a value greater than 1e-3. Using the flag `-eps VAL`, we can specify a different threshold for the classifier. 

## Example run
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

# Usage with NVDLA

## Folder structure for NVDLA results
NVDLA quantizes all tensors; "bitwidth" refers to the quantization precision adopted for the experiments (e.g., int8).
"config" refers to the accelerator configuration used to run the experiment.
All errors must be provided in .npz archives, each containing a sequence of .npy files. Each .npy file is a 5D tensor: the first dimension is the error group, the second is the batch, and the remaining three are the proper corrupted tensor.

```
network1/
├── layers/
|   ├── bitwidth1/
|   |   ├── layer1/
|   |   |   └── golden.npy
|   |   ├── ...
|   |   └── layerN/
|   ├── ...
|   └── bitwidthN/
├── config1/
|   ├── layer1/
|   |   ├── ctrl/
|   |   |   ├── unit1/
|   |   |   |   └── errors.npz
|   |   |   ├── ...
|   |   |   └── unitN/
|   |   └── data/
|   |       ├── unit1/
|   |       |   └── errors.npz
|   |       ├── ...
|   |       └── unitN/
|   ├── ...
|   └── layerN/
├── ...
└── configN/
```

## Running the tool (NVDLA)
```bash
python src/main_nvdla.py <layer_folder> <relative_golden_tensor_path> <relative_errors_path> <output_folder> -as --classes <model_name> '' [options]
```

As an example, suppose to have operator folder `conv1` structured as explained above. In that case, the command would be

```bash
python src/main_nvdla.py </path/to/conv1> ../golden.npy errors.npz </path/to/output/folder> -as --classes conv1 '' [options]
```

### Options (NVDLA)
The following options are the most relevant ones for an NVDLA injection campaign. Run the command with the `--help` flag to see a complete list.

* #### **Visualization**
    This tool is capable of creating visualizations of the identified errors. Adding the flag `-v` or `--visualize` will enable this functionality. The images produced will be grouped together according to the underlying spatial pattern. 
    N.B. creating such a visualization is costly and will make the execution of the tool slower.
* #### **Parallelism**
    To speed up the execution of the tool, it is possible to enable multiprocessing. To do so, use the flag `-p N`, which will spawn `N` threads working in parallel. 
* #### **CLASSES Models**
    The goal of performing fault injections is to create error models that CLASSES can use. This tool can make the required JSON files during the analysis to aid this process. To enable this process, use the flag `--classes Sx Operator`, where Sx is the number of the experiment, and Operator is the name of the currently analyzed operator. I.e., if you are creating the 4th model for the convolution, use the flag `--classes S4 Conv`. NOTE: This naming convention is deprecated, but two string arguments after `--classes` are still required. You do not have to follow striclty the naming convention. If the flag is `--classes A B` the file will be named `A_B.json`.
* #### **Error models for each hardware unit**
    If you wish to study the behavior of a single hardware unit, use option `--classes-unit-models`. This will generate one error model per hardware unit.
    N.B. This requires the `--classes` option to be enabled, otherwise model generation will fail.
* #### **Epsilon**
    By default, this classifier considers an error in each value that differs from the golden version by a value greater than 1e-3. Using the flag `-eps VAL`, we can specify a different threshold for the classifier.
* #### **IMPORTANT: Almost-same**
    The -as or --almost-same option MUST be used to properly use epsilon in tensor value inequalities. If you omit it, the classifier will instead simply check whether the values are different.

## Channel counting (NVDLA)
Directory `channel_counting` contains some scripts to perform an additional check: for each corrupted tensor in the results of a specified layer, this tool determines the spatial class of the tensor and counts the number of corrupted channels in it, reporting the results in a `channel_counts.csv` file for each HW unit. These results are then aggregated (also for each unit) in a `class_frequencies.csv` file, listing the percentages of single-channel and multi-channel tensors encountered for each spatial class.

The counting tool is run mostly like the classifier:
```bash
python src/channel_counting/count_class_channels.py <layer_folder> <relative_golden_tensor_path> <relative_errors_path> <output_folder> [options]
```
The only two available options are -p and -eps.

Be aware that this tool runs a simplified version of the classifier; while it is faster, it still iterates over all corrupted tensors of the given layer, meaning that it may take a significant time.

## Extending the set of recognized spatial classes
Note that the following is not an exhaustive guide, but simply a collection of pointers that may be followed to more easily add new spatial classes to the set of recognized ones.

First, implement a recognizer function in a new script in `src/spatial_classifier/classifiers`. It is suggested that you name the script after the new spatial class. Take a look at `template.py` to see the general structure of a recognizer function.

Next, extend the Enum defined in `src/spatial_classifier/spatial_class.py` by adding an entry for your new class; also add an entry to the `to_classes_id()` function below the Enum definition.

In `src/spatial_classifier/spatial_classifier.py`, extend the import list at the top by importing your new recognizer function. Then, add an entry to either the `SINGLE_CHANNEL_CLASSIFIERS_NEW` dictionary (if you new spatial class only affects one channel) or the `MULTI_CHANNEL_CLASSIFIERS_NEW` dictionary (if multiple channels are affected). The entry is a pair, associating the new Enum entry you defined with the new recognizer function.
These two dictionaries are iterated over by the classifier and each entry's function is used to test the corresponding spatial class against a corrupted tensor. If you want the classifier to skip some spatial classes, simply comment out the related entries in the relevant dictionary.

These steps should cover the majority of the classifier extension process. If the new recognizer function does not work straight away, you may want to check the other scripts in the `src` directory, starting from `main_nvdla.py` and possibly focusing especially on `batch_analyzer_nvdla.py` and `tensor_analyzer.py`, which implement most of the classifier's logic.

## NVDLA Example Run
Download an example NVDLA FI output [here](https://miele.faculty.polimi.it/classifier_example_data.tar.xz).

Move it to the root folder of the classifier and extract with:
```
tar -x -I xz -f classifier_example_data.tar.xz
```

The extracted directory contains an `alexnet_cifar10` subdirectory, which contains another subdirectory related to NVDLA's 8x8_int8 configuration; the latter contains the results for the first three convolutional layers of AlexNet. This example uses the first layer; you may want to use the other two for additional practice.

To run classification on layer conv1, execute the following commands from the root directory of the classifier:
```
CONV1_DIR="alexnet_cifar10/nv_8x8_b1_dat-524288_wt-32768_int8/conv1"

python src/main_nvdla.py ${CONV1_DIR} ../../golden.npy ./errors.npz ${CONV1_DIR}/classes -as --classes conv1 '' --classes-unit-models -p 4
```
Since classificaton can be lengthy, you may want to run the last command in the background by appending `&` to it. If you're connected remotely and want the process to keep running even if connection drops, add `nohup` to the start, before `python`.

The individual unit error models, as well as the overall layer error model (called `conv1_.json`), will be generated in the `classes` directory within `conv1`.

Now run the channel counting procedure with
```
CONV1_DIR="alexnet_cifar10/nv_8x8_b1_dat-524288_wt-32768_int8/conv1"

python src/channel_counting/count_class_channels.py ${CONV1_DIR} ../../golden.npy ./errors.npz ${CONV1_DIR}/classes -p 4
```
This will create the `class_frequencies.csv` in the `conv1` directory.
# cnn-error-classifier

Compares corrupted tensors with a gold one, classifying the differences by their domain and their spatial distribution.

## Features

The input of the program is a golden .npy tensor and a folder containing faulty .npy tensors (both NHWC or NCHW formats can be processed).

The program will classify the faulty tensor based on the spatial distribution of the erroneous values. For each tensor an image
that shows visually the differences with the golden tensor will be generated.

The program also generates a json report containing info about the analysis 

## Limitations

Only tensors with batch dimension (N) equals to 1 can be processed.

## Setup
The program is tested from python > 3.6

Create a virtual environment with your favorite package manager.

Install the dependencies from the requirements.txt file

* Venv:
```sh
pip install -r requirements.txt
```

* Conda:
```sh
conda install --file requirements.txt
```

## Usage

To run the program use:
```sh
python src/classifier.py [options] <path to golden npy tensor> <path to folder with corrupted tensors> <path to analysis output folder>
```
It is possible to specify -nhwc to process tensors that are stored using the NHWC format 

To see what options are available use
```sh
python src/classifier.py -h
```

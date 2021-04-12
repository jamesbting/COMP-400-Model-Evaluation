# COMP-400-accuracy-calculator
A script to compute the mean win rates for the recommendations

This program uses a trained neural network to compute the probability of winning for the blue team. It then takes all recommendations and computes the mean probability of winning for the blue team. 

## Installation

Before running the program, ensure you have the following dependencies installed. This program is not validated on any other version of the dependencies. 

- Python 3.9.2
- PyTorch 1.7.1

## Before running

Before running the program, ensure that the file locations are as desired. The program has configuration options:

- model: A string representing the path to the desired model. The model should be a PyTorch neural network stored as a ```.pickle``` file.
- results_files: A list containing strings of paths to all the files with the recommendations. Each file should be a csv, where each row contains 10 numbers, representing the champions selected by each player.

## Running the program

To run the program, go to the location that the repository is located, and issue the following command:

```
python src/main.py
```

The mean win rate for the blue team will be printed to STDOUT as a result. 
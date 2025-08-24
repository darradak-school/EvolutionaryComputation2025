# Evolutionary Computation 2025 - Assignment 1

Code and documentation for Assignment 1 of the Evolutionary Computation 2025 course.

## Contents

- `code/` - Source code for the assignment
- `doc/` - Documentation files as per assignment specification
- `results/` - Results files

## Getting Started

1. Clone the repository / Download the files.
2. Follow instructions in the documentation to run the code.

## Requirements

- Python 3.x
- numpy installed on system

## Usage
Run local search algorithm - Ex.2
"python localsearch.py"
Can configure localsearch variables in the main function to change the algorithm parameters. 
E.g. add new problems, alter cooling rate, change the stagnation limit, etc.

Run evolutionary algorithm testing and benchmarking Ex.6
"python evolutionary_modular.py"

(Would take >24hrs to complete USA and pr2392 on larger test cases - ran on smaller test examples even after multithreading/parellelizing due to submission deadlines)
(Added parellelizisation may cause issues on some systems but should work naturally using up to as many processes as possible)

Run inverover algorithm - Ex.7
"python inverover.py"
Can configure inverover variables in the main function to change the algorithm parameters.
E.g. add new problems, alter population size, change inversion probability, etc.
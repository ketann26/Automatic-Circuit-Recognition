# Circuit-Solver

## Objective
***This project aims to create LTspice schematic from hand-drawn circuits for the students who don’t know the interface of simulators***
## How it Works
***It will recognize each component of a circuit and create its corresponding schematic.<br/> We have used segmentation using an algorithm for identification of open line elements capacitor, battery, ground, resistor etc.***
## Techstack
***Machine Learning is being used for identification of resistors and other continuous elements and Deep Learning for the identification of the values associated with each component.***
## Execution
***The user has to  scan the hand-drawn circuit image and then with the help of our model, we recognize the different components present in the circuit and simulate the same with the help of LTSpice Python API to generate the output waveform/graph.***

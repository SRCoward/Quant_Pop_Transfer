# Quant_Pop_Transfer
MPhil Mini Project 2: Simulating a Quantum Algorithm for Population Transfer via Non-Ergodic Extended States for Molecular Dynamics

Here is where I shall develop the second mini project on using the quantum population transfer protocol as a subprocess of a quantum parallel tempering algorithm. 

Main Reference: https://arxiv.org/pdf/1807.04792.pdf

The report should contain the following:

I. An introduction to quantum computing and its applications in chemistry, as well as comments on the current capabilities of existing hardware. 						       [15%] 

II. A description of the Population Transfer protocol and how it can be used as a subroutine for quantum parallel tempering, as well as a description of the Trotter decomposition techniques used  to simulate the quantum algorithm.								        [15%]

III. Results will be presented for the technique applied to a particular problem Hamiltonian, which includes pairwise interactions, to be selected upon further analysis of the scope of the algorithm. We will use histograms to compare the frequency with which energy eigenstates are located via PT and alternative classical techniques. 								        [30%]

IV. Conclusions and future outlook								        [10%]

V. List of References										          [5%]

## Code Overview
Includeded in this repository are two evolution methods for simulating the population transfer protocol applied to the 2-local Hamiltonian, given in the report, and also the code used to generate an example of the landscapes we investigate. 
1. IB_hamiltonian - contains a basic python script to generate the example landscape plot given in the report.
2. Circuit Simulation - combines the the qiskit python package to generate the circuit and QuEST simulator to compute the results of our circuit. 
3. Classical Simulation - combines a python script which sets up the problem and computes the eigenvalues of the Hamiltonian. The C++ script then evolves the state using Trotter Decomposition as referenced in the project report. 

## IB Hamiltonian
A pair of scripts which compute a simple Hamiltonian as described by Kechedzhi in the reference above. Running IB_landscape_plot.py will generate a plot of energies against state number. 

## Circuit Simulation
The circuit simulation code requires the installation of additional packages and complilation of the quantum simulator. 
Firstly qiskit and qiskit aqua (the algorithm package) can be installed using the pip tool.

```bash
pip install qiskit qiskit-aqua
```
For further details please visit: https://github.com/Qiskit/qiskit-aqua.

QuEST, the simulator, is simply a folder of .c and .h files therefore only needs compilation. To download QuEST follow the instructions at https://quest.qtechtheory.org/download/, then ensure the makefile has the correct location to your QuEST directory as this will vary according to your setup. The following will download the necessary files.
```bash
git clone "https://github.com/QuEST-Kit/QuEST.git"
```
Running make then execute_circuit_simulation will allow you to specify the size of system and will also ask you to specify the initial state which will be output by the python script. The python script initially generates the QASM file, which describes the circuit, then the C++ file simulates and measures the outcome. Two plots are generated, one of energy against frequency, and one of the hamming distance from the initial state against frequency.

## Classical Simulation
A combination of a python script to generate the eigenvalues, followed by a C++ script to actually implement the evolution of the system. 

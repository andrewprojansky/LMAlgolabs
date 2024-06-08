![Banner](banner.png)

# AlgoLab Summer School 2024
# Tutorial on Simulating Quantum Many-Body Systems with Language Models

This tutorial gives some examples on using language models as a wave function ansatz for quantum many-body systems.
In our scenario, we combine a variational Monte Carlo approach with recurrent neural networks (RNNs) and transformer models to find ground state representations in 2D Rydberg atom arrays.

The following resources were consulted for this tutorial. You can consult them for further knowledge:

- [Sprague and Czischek, 2024](https://www.nature.com/articles/s42005-024-01584-y)
- [Zhang and Ventra, 2023](https://physics.paperswithcode.com/paper/transformer-quantum-state-a-multi-purpose)
- [Czischek et. al., 2022](https://arxiv.org/pdf/2203.04988)
- [Hibat-Allah et. al., 2020](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358)
- [Deep Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)
- [QuCumber](https://github.com/PIQuIL/QuCumber)

With permission, code in the [APRIQUOT](https://github.com/APRIQuOt/VMC_with_LPTF) repository was used in the transformer training.

## Considered Physical System

We are considering a two-dimensional array of Rydberg atoms on a square lattice.
Those are described by the Rydberg Hamiltonian:
```math
\hat{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right)
```
with $`\hat{\sigma}_i^x=\left|\mathrm{g}\right\rangle_i\left\langle\mathrm{r}\right|_i+\left|\mathrm{r}\right\rangle_i\left\langle\mathrm{g}\right|_i`$ and $`\hat{n}_i=\left|\mathrm{g}\rangle_i\langle\mathrm{g}\right|_i`$. Here $`\left|\mathrm{g}\right\rangle`$ and $`\left|\mathrm{r}\right\rangle`$ denote the ground and excited (Rydberg) state, accordingly.

## Goals

The learning objectives include

- Seeing a language model as a variational wave function ansatz to search for ground state
- Learning how the Hamiltonian of the problem slots into a machine learning framework. Our loss function will be the expectation value of the Hamiltonian
- Using a trained network to compute observables of a system

## Models

- Recurrent Neural Network (Gated Recurrent Unit)
- Transformer (with Multihead Self-Attention)

## Language/Framework

- [Python3](https://www.python.org)
- [Jax](https://jax.readthedocs.io)
- [Flax](https://flax.readthedocs.io)
- [Torch](https://pytorch.org)

## Requirements

You should have both of the following installed on your local machine one way or the other

- [Python3](https://www.python.org)
- [Git](https://git-scm.com/) (You may need to download the folders as a compressed folder if you do not have git installed)

# Part 1: Getting to know the tools
In this first part, we will explain the considered physical model in more detail and see how language models are implemented and used for quantum state representations.
You can use the following steps depending on your operating system to get the tutorial running.
Those steps will open an application which will guide you through the first part of the tutorial.

## How to Use

Follow these instructions for usage. [Max/Unix](#unixmac), [Windows](#windows), [Makefile](#advanced-users-makefile)

### Unix/Mac

1. Clone the repository:

    ```bash
    git clone https://github.com/lere01/tutorial_nqs.git
    ```

2. Change directory:

    ```bash
    cd tutorial_nqs
    ```

3. Run the setup script:

    ```bash
    bash run.sh
    ```

Remember to run `chmod +x run.sh` to make the script executable before running. Some unix based systems allow you to simply double-click on the file.

### Windows

1. Clone the repository:

    ```bash
    git clone https://github.com/lere01/tutorial_nqs.git
    ```

2. Change directory:

    ```bash
    cd tutorial_nqs
    ```

3. Run the setup script:

    ```bat
    run.bat
    ```

Note that opening the root directory in Windows Explorer and double clicking `run.bat` can also achieve the same thing.

### Advanced Users (Makefile)

The advantage of the Make commands is the fine grained control you get over running/stopping the app and cleaning your environment. So if you already have `make` setup on your PC/Mac, then using the following commands would serve better.

1. Run the setup and start the application:

    ```bash
    make run
    ```

2. To stop the application:

    ```bash
    make stop
    ```

3. To clean up the environment:

    ```bash
    make clean
    ```

These steps will ensure that the virtual environment is created, dependencies are installed, and the application is run, all with a single command, making it easier for you to get started.

# Part 2: Evaluating Observables

The second part of the tutorial is provided as a Jupyter notebook.
To access the notebook, use the following steps depending on your operating system:

### Unix/Mac
1. Activate the virtual environment:
   ```source venv/bin/activate```
2. Open the notebook:
   ```jupyter notebook```
   Then navigate to the ```notebook``` folder and open the notebook ```tutorial_2.ipynb```.
   
### Windows
1. Activate the virtual environment:
   ```venv\Scripts\activate```
2. Open the notebook:
   ```jupyter notebook```
   Then navigate to the ```notebook``` folder and open the notebook ```tutorial_2.ipynb```.

In this part of the tutorial, you will write functions to evaluate observables using samples from a trained language model.
To make the code development more efficient, the tutorial will load a pre-trained RNN from which samples are generated to evaluate observables.
The notebook will guide you through the individual steps of this second part of the tutorial.

# Part 3: Getting Experience

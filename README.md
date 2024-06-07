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
$$
\tilde{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right )
\end{equation}
    \tilde{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right ) 
$$

Let us consider the physics of the problem.

- We are looking at a 2D square lattice of Rydberg atoms
- We will be using the Rydberg Hamiltonian

## Goals

The learning objectives include

- Seeing a deep learning model combined with variational ansatz to search for ground state
- Learning how the Hamiltonian of the problem slots into a machine learning framework. Our loss function would be the expectation of the Hamiltonian
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

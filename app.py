import startup ## this has to be the first import
import os
import streamlit as st


st.set_page_config(
    page_title="Welcome - NQS Tutorial",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown('<link href="static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True)
st.title("Neural Networks for Wave Functions Parameterization")

cwd = os.getcwd()
image_path = os.path.join(cwd, "static", "nn_models.png")
st.image(image_path)

# Body Section
st.markdown(
    r"""
        ## Welcome
        This app allows you to explore the parameterization of wave functions using language 
        models as artificial neural networks. This tutorial will introduce 
        you to the idea of using language models as a wave function ansatz to represent quantum 
        states. In our scenario, we combine a variational 
        Monte Carlo approach with neural quantum state techniques to search for the ground 
        state of a 2D lattice of Rydberg atoms.

        ## Acknowledgements

        - The above image and other images in other pages were taken from [Sprague and Czischek](https://www.nature.com/articles/s42005-024-01584-y/figures/1). 
        - The following resources were consulted for this tutorial
            - [Sprague and Czischek, 2024](https://www.nature.com/articles/s42005-024-01584-y)
            - [Zhang and Ventra, 2023](https://physics.paperswithcode.com/paper/transformer-quantum-state-a-multi-purpose)
            - [Czischek et. al., 2022](https://arxiv.org/pdf/2203.04988)
            - [Hibat-Allah et. al., 2020](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358)
            - [Deep Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html)
            - [QuCumber](https://github.com/PIQuIL/QuCumber)

        - With permission, code in the following repository was used for the transformer:
            - https://github.com/APRIQuOt/VMC_with_LPTF

        ### Physics of the Problem

        Let us consider the physics of the problem.

        - We are looking at a 2D square lattice of Rydberg atoms
        - Rydberg atoms can be prepared in a ground and a highly excited (Rydberg) state and we can consider them as qubits
        - We are assuming all-to-all interactions among lattice sites
        - The Rydberg Hamiltonian is as follows
        $$
        \begin{equation}
            \hat{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right )
        \end{equation} 
        $$

        with the van der Waals interaction  $V_{ij} = \frac{\Omega R_{\mathrm{b}}^6}{| \textbf{r}_i - \textbf{r}_j |^6}$. $R_{\mathrm{b}}$ is the Rydberg blockade radius within which any two excitations are penalized.

        - $\Omega$ is the Rabi frequency describing Rabi flops between the two states
        - $\delta$ is the detuning of the Rydberg state
        - $\hat{\sigma}_i^x=\left|\mathrm{g}\right\rangle_i\left\langle\mathrm{r}\right|_i+\left|\mathrm{r}\right\rangle_i\left\langle\mathrm{g}\right|_i$ is the Pauli-$X$ matrix acting on qubit $i$
        - $\hat{n}_i=\left|\mathrm{g}\right\rangle_i\left\langle\mathrm{g}\right|_i$ is the number operator acting on qubit $i$
        - Atoms at positions $\textbf{r}_i$ and $\textbf{r}_j$ interact through the van der Waals potential, $V_{ij}$
        - $N=N_x\times N_y$ is the total number of lattice sites, where we look at square lattices with $N_x=N_y$
        
        Note that we set $\Omega = \delta = 1$ and $R_b = 7^{\frac{1}{2}}$ as default choice in this tutorial. This brings the system in the vicinity of transition between the disordered 
        and the striated phase.

        ### A Bird's Eyeview of the Approach
        - Step 1: Parameterize a wave function with an ansatz (language models in our case)
        - Step 2: Sample from the squared wave function amplitude
        - Step 3: Compute the expectation value of the energy
        - Step 4: Vary the variational parameters using an optimization (loss) function
        - Repeat Steps 2-4 until convergence is reached
        - To converge to the ground state, the loss function is the energy expectation value of the system
    """
)

# Next Pages
st.markdown(
    r"""
        HAVE FUN!
    """
)

# Initialize Session State
if "model_config" not in st.session_state:
    st.session_state.model_config = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "vmc_config" not in st.session_state:
    st.session_state.vmc_config = None

# Footer Navigation
# Add vertical space
for _ in range(5):
    st.write("")

_, _, _, _, _, _, _, col1 = st.columns(8)

# go to home page if clicked
with col1:
    st.page_link("pages/configuration.py", label="Get Started", icon=":material/arrow_forward:")







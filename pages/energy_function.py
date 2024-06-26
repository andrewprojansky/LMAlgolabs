import io
import sys
import os
import ast
import copy
import json
import textwrap
import numpy as np
import streamlit as st
from typing import List
import jax.numpy as jnp
from jax import lax, random
import matplotlib.pyplot as plt
from code_editor import code_editor
from streamlit_extras.let_it_rain import rain
from streamlit_extras.switch_page_button import switch_page

from src.rnn_model.vmc import VMC
from src.helpers import ModelType, run_tf_model, fake_logpsi, check_flip_state, check_transverse_fn, LineCollector, meets_cond, save_to_file

st.set_page_config(
    page_title="Energy Function - NQS Tutorial",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "model" not in st.session_state:
    switch_page("app")


cwd = os.getcwd()
button_file_path = os.path.join(cwd, "static", "buttons.json")
with open(button_file_path, "r") as file:
    buttons = json.load(file)

st.markdown(
    '<link href="../static/css/styles.css" rel="stylesheet">', unsafe_allow_html=True
)
st.title("Energy Function")



# Provide a code editor for the user to edit the code
st.markdown(
    r"""
    ### Compute the Local Energy
    Are you ready for some tasks? 

    - Replace CHANGE_ME in function `step_fn_chemical`. 
    - Complete the implementation of the `flip_state` function.
    - Complete the implementation of the `step_fn_transverse` function.
    - Complete the expression for `loc_e`
    
    When you are done, click the `Run Code` button or press `Ctrl + Enter` (`command + return` for Mac) to run the code. Use the Hamiltonian 
    shown below as guide

    $$
        \begin{equation}
            \hat{H} = - \frac{\Omega}{2} \sum_{i = 1}^N \left( \hat{\sigma}_i^x \right) - \delta \sum_{i = 1}^N \left ( \hat{n}_i \right ) + \sum_{i,j} \left ( V_{ij} \hat{n}_i \hat{n}_j \right )
        \end{equation} 
    $$

    - :red[**WARNING**]: Only change the places where you find `CHANGE_ME`. Changing any other part of the code may result in an error.
    - :blue[**HINT**]: Remember that `delta` and `Omega` are the detuning and Rabi frequency respectively. They play crucial roles in the computation of the local energy.
   - :red[**HINT**]: If you're training a transformer model, an output folder system called "TF" is created. The folder with the highest number is your most recent model. You can find the energy expectation value for all training steps in the file "DEBUG.npy", which contains an array that can be loaded into Python. The energy expectation value is stored in the first column of this array. 
    """
)

boilerplate_code = textwrap.dedent(
    """
    # Import Libraries
    import numpy as np

    # Define the Local Energy
    def local_energy():
        '''Compute the local energy of the wave function.'''
        print('Hello World!')

    local_energy()

    print('Completed!')
    """
)

runner_code_file = os.path.join(cwd, "src", "rnn_model", "runner.py")
with open(runner_code_file, "r") as file:
    runner_code = file.read()
run_model_code = textwrap.dedent(runner_code)
run_model_globals = {
    "vmc_config": st.session_state.vmc_config,
    "model": st.session_state.model,
    "VMC": VMC,
    "jnp": jnp,
    "np": np,
    "lax": lax,
    "List": List,
    "rng_key": random.PRNGKey(1234),
    "st": st,
    "io": io,
    "sys": sys,
    "copy": copy,
    "delta": 1.0,
    "Omega": 1.0,
    "state": st.session_state,
    "params": .00005,
    "model": .00007,
    "get_logpsi": fake_logpsi,
    "log_psi": .00002,
}

# Display the code editor
completed_code = code_editor(
    run_model_code,
    lang="python",
    theme="contrast",
    height=300,
    buttons=buttons,
    key="run_model_code",
)

def energy_plot(densities):
    fig, ax = plt.subplots()

    ax.plot(densities, label='Energy Density', color='g')
    ax.set_ylabel('Energy Density')
    ax.set_xlabel('Training Epoch')
    ax.set_title('Energy Density Over Training Epoch')
    ax.hlines(-0.45776822, 0, 1000, colors='r', linestyles='dashed', label='Target Energy Density')
    ax.legend()

    st.pyplot(fig, use_container_width=False)


if completed_code["type"] == "submit":
    tree = ast.parse(completed_code["text"], type_comments=True, mode="exec")
    tree_code = compile(tree, "tempp", mode = "exec")
    exec(tree_code, run_model_globals)

    
    with st.spinner("Checking your solution..."):
        # initialize task statuses
        todo_1_correct = False
        todo_2_correct = False
        todo_3_correct = False
        todo_4_correct = False

        # extract the code from the code editor
        code = completed_code["text"]
        code_lines = code.strip().split('\n')


        # Step_fn_chemical
        # Check if the user has completed the first todo
        todo_1 = code_lines[10].strip().split()

        if todo_1[3] != 'delta':
            st.write("You might want to check on ToDo_1")
        else:
            todo_1_correct = True
            st.write("Well done! You have successfully completed the todo 1.")

        # Flip_state function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "flip_state":
                func_def = compile(ast.Module(body=[node], type_ignores=[]), '', 'exec')
                exec(func_def, run_model_globals)
            
            if isinstance(node, ast.FunctionDef) and node.name == "step_fn_transverse":
                func_def = compile(ast.Module(body=[node], type_ignores=[]), '', 'exec')
                exec(func_def, run_model_globals)
            
            

        if check_flip_state(run_model_globals["flip_state"]):
            todo_2_correct = True
            st.write("Well done! You have successfully completed the todo 2.")
        else:
            st.write("You might want to check on ToDo_2")


        # Step_fn_inverse
        # Check if the user has completed the third todo
        it_works = check_transverse_fn(run_model_globals["step_fn_transverse"])
        if it_works:
            todo_3_correct = True
            st.write("Well done! You have successfully completed the todo 3.")
        else:
            st.write("You might want to check on ToDo_3")


        
        # Check loc_e implementation
        collector = LineCollector()
        collector.visit(tree)
        target_line = code.splitlines()[collector.target_line - 1]
        loc_e_done = meets_cond(target_line)

        if loc_e_done is True:
            todo_4_correct = True
            st.write("Well done! You have successfully completed the todo 4.")
        else:
            st.write("You might want to check on ToDo_4")


        # Display checkboxes
        st.subheader("Checklist")
        st.checkbox("TODO_1: step_fn_chemical correctly implemented", value=todo_1_correct, disabled=True)
        st.checkbox("TODO_2: flip_state correctly implemented", value=todo_2_correct, disabled=True)
        st.checkbox("TODO_3: step_fn_transverse correctly implemented", value=todo_3_correct, disabled=True)
        st.checkbox("TODO_4: Total energy computation correctly implemented", value=todo_4_correct, disabled=True)


    # Execute the code
    if all([todo_1_correct, todo_2_correct, todo_3_correct, todo_4_correct]):
        rain(
            emoji="🎈",
            font_size=54,
            falling_speed=5,
            animation_length="10s",
        )

        # initialize file path
        file_path = os.path.join(st.session_state.config_dir, "energy_densities")
        if st.session_state.model_type.name == ModelType.RNN.name:
            try:
                with st.spinner("Your RNN model is training..."):
                    # exec(completed_code["text"], run_model_globals)
                    my_vmc = VMC(
                    nsamples = st.session_state.vmc_config.n_samples,
                    n= st.session_state.vmc_config.nx,
                    learning_rate = st.session_state.vmc_config.learning_rate,
                    num_epochs=st.session_state.vmc_config.num_epochs,
                    output_dim=st.session_state.vmc_config.output_dim,
                    sequence_length=st.session_state.vmc_config.sequence_length,
                    num_hidden_units=st.session_state.vmc_config.num_hidden_units
                    )

                    # Initialize the model
                    dummy_input = jnp.zeros((st.session_state.vmc_config.n_samples, st.session_state.vmc_config.sequence_length, st.session_state.vmc_config.output_dim))
                    params = st.session_state.model.init(random.PRNGKey(1234), dummy_input)
                    e_den = my_vmc.train(random.PRNGKey(123), params, st.session_state.model)

                    st.session_state.densities = [(i.mean() / st.session_state.vmc_config.sequence_length).item() for i in e_den]

                    save_to_file(
                        filename=file_path, 
                        obj=st.session_state.densities,
                        save_type=st.session_state.save_type
                    )

                    # with open("reults.pkl", "wb") as f:
                    #     pickle.dump(st.session_state.densities, f)

                    st.session_state.training_completed = True

                energy_plot(st.session_state.densities)
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            try:
                with st.spinner("Your Transformer model is training..."):
                    run_tf_model(st.session_state.model, st.session_state.full_opt, st.session_state.opt_dict)

                    st.session_state.training_completed = True

            except Exception as e:
                st.error(f"Error executing code: {e}")


        rain(
            emoji="🎈",
            font_size=100,
            falling_speed=10,
        )
        st.success("CONGRATULATIONS! YOU COMPLETED THE MODEL TRAINING")
        st.write("You can use the saved configuration/model to compute observables in the next exercise.")


# Footer Navigation
col1, _, _, _, _, _, _, col2 = st.columns(8)

with col1:
    st.page_link(
        "pages/model_confirmation.py", label="Back", icon=":material/arrow_back:"
    )


with col2:
    st.page_link("app.py", label="Restart", icon=":material/refresh:")

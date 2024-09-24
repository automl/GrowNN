# Master Thesis: Growing With Experience; Growing Neural Networks in Deep Reinforcement Learning
This repository holds the code for the Master Thesis: `Growing With Experience; Growing Neural Networks in Deep Reinforcement Learning`.

In the following we first provide an installation guide, to then explain the repositories structure, and lastly explain how experiments can be run.

## Installation
1. To install our environemnt, you first need to clone our implementation and all deendecies using
```bash
git clone https://github.com/automl-private/architectures-in-rl.git --recursive
```
2. Then, create a new anaconda environemnt, possibly using
```bash
conda create -n network_growing python=3.8.19
```
3. Next, install the nethack learning environment. If you encounter issues refer to the documentation at https://github.com/facebookresearch/nle
```bash
pip install ./nle
```
4. To install py_experimenter you first need to adapt `./py_experimenter/py_project.toml` to 
```toml
[tool.poetry.dependencies]
python = "^3.8"
```

5. Next, install MiniHack, PyExperimenter, StableBaselines3 and SMAC3 using 
```bash
pip install ./minihack
pip install ./SMAC3
pip install ./py_experimenter
pip install ./stable-baselines3
```

6. Lastly install the local repo and dependencies using
```bash
pip install .
pip install -r requirements.txt
```

## Code Structure

The main codebase is organized into the following directories:

- **approach**: Contains the core files for running experiments related to different approaches.
    - **approach/ant**: Holds the files for running experiments on the Ant environment.
    - **approach/minihack/net2deeper**: Contains files to run Net2Deeper experiments for a fixed MiniHack environment.
    - **approach/minihack/net2wider**: Contains files to run Net2Wider experiments for a fixed MiniHack environment.
    - **approach/minihack/increase_difficulty**: Contains files to execute Net2Deeper experiments, where the MiniHack environment increases in difficulty as the agent is trained.
    
- **baselines**: Stores the run files for baseline experiments, structured similarly to the `approach` directory.

- **config**: Shold ontains the configuration files for `py_experimenter`.

- **hydra_plugins**: Contains early versions of the hypersweeper code.

- **plotting**: Includes all scripts and notebooks used for generating plots.

- **test**: Holds elementary test cases for various network morphisms.

- **utils**: Contains network implementations and additional utility files.

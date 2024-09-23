# Currently Testing Ground 

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
conda create -n network_growing python=3.8.19
```

4. To install py_experimenter you first need to adapt `./py_experimenter/py_project.toml` to 
```toml
[tool.poetry.dependencies]
python = "^3.8"
...
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

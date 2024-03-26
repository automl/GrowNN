# Currently Testing Ground 
We aim to find out whether or not one can adapt the architecture of the actor and critic in a popular stable-baselines3 library.

## Installation
To make the code runnable, you first need to create and activate an environment using
```bash
conda create -n growing-nn
conda activate growing-nn
```
followed by an installation of `Python` in the environment
```bash
conda install python==3.9
``` 

Then you need to install the dependencies. 

### NETHACK
Note that you need to install the NetHack library `nle` first. To do so follow [the official documentation](https://github.com/facebookresearch/nle?tab=readme-ov-file#installation). Note that the library is only runnable on `Linux` and `MAC` and requires both `GCC` and `CMAKE` (>=3.15). On wsl this works best if installed using Anaconda with:
```bash
conda install gcc
conda install cmake=3.26.4
```

### General Dependencies
The rest of the dependencies can be installed using 
```bash
pip install -r requirements.txt
```

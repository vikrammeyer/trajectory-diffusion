# diff traj

- `chmod +x env.sh` and then `./env.sh` to setup the virtual env.

- `pip3 install -e .` to install the project pkg locally.

- Use the following snippet in Jupyter notebooks to auto reload any changes to your package:

```
%load_ext autoreload
%autoreload 2
from pkg.module import Class, function
```

## additional setup on VM
- `sudo apt-get update`
- `sudo apt-get install git-all`
- `sudo apt-get install python3-venv`
- `sudo apt-get install zip`

To move folders, use `zip -r filename.zip foldername` and then download the file using the google ssh in browser utilities.

## experiments
- Diffusion:
  - Unet1D states (1 channel)
  - Unet1D states (4 channels- x, y, theta, v)
- Baselines (implement in this priority order in case we don't have time for it all):
  - (1) Fully Connected: obstacles -> full state trajectory
  - (2) RNN/LSTM: many to many
  - (3) CVAE: state trajectory, conditioned on obstacles

### wont do:
- dynamic obstacles
- Unet1D states & controls as separate information channels (nvm won't work b/c dimensions won't match)

## metrics
- % collision free (num of states collision free / total)
- dynamics violations
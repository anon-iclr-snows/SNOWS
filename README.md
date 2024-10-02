This repository contains the implementation of the SNOWS. The code implements the layer-wise pruning pipeline and conjugate gradient methods described in our main paper. 

## Repository Structure
- `./model/`: Contains the architectures currently supported by SNOWS.
- `./prune/`: Contains the SNOWS layer-wise pruning pipeline (Algorithms 2 & 3 in the main paper).
- `./SNOWS/`: Contains the SNOWS Conjugate Gradient (CG) code (Algorithm 1 in the main paper).
- `run.py`: Python script used to execute the layer-wise pruning pipeline.
- `run.sh`: Shell script to run `run.py` with specified arguments.

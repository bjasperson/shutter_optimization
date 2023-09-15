# Shutter optimization code and datasets.

## Setup
- [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repo to your machine
- Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the attached environment.yml file 

## Python modules
- image_creation.py: generates random image designs for data generation. Also can create checkerboard pattern and simulated results for full images.
- data_merge.py: combine simulation results into readable format for neural network. Also can analyze/plot results. 
- pixel_nn.py: train/testing for perf_net.
- perf_net.py: normalize images and results, rescale results, and perf_net architecture.
- pixel_optim_nn.py: train/testing for top_opt.
- plotting.py: plotting functionality for paper.
- studies.py: creation of design front (Pareto figure), along with hyperparameter optimization

## Training Data
- combined_results_original: original dataset used for optimization; temperature in K.
- combined_results_dT: same dataset as original, but updated temperature to report dT.
- comsol_sim.mph: comsol file for training data generation.
- raw_data.tar: output files from comsol simulations (to be combined using data_merge.py).

# Learned Weight Sharing

This is the code release for the paper: Prellberg J., Oliver K. (2020)
Learned Weight Sharing for Deep Multi-Task Learning by Natural Evolution
Strategy and Stochastic Gradient Descent. Unpublished.

## Usage

Use the script `run_training.py` to train a model. Every dataset that can be
trained on has its own directory containing a `dataset.py` and `model.py` file.
After selecting a dataset via `--dataset` the `--model` option refers to
a symbol in the `model.py` file. The different cases described in the paper
(full sharing, no sharing, learned sharing) are all available as different
models.

Instead of starting the training directly, it can be easier to use the scripts
in the `launchers` directory. Starting one of the scripts without any arguments
will print a numbered list of configurations. Start a script with an integer
argument to select one of the configurations to execute.

During the training, summary data will be written to `npz` files. Their
contents can be read using the `run_inspect_file.py` and `run_plot.py` scripts.

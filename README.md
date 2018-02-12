# README

To run the basic MNIST example, do `python main.py`.

## Dependencies
* pytorch
* tensorboardX
* tensorflow

## Visualizing results
run `tensorboard --logdir runs`

## Examples

### Using `main.py`

`python main.py --tensorboard-plot=test --log-output --batch-size=1000`

This adds a run called `MNIST_0.01_1000_0` (learning rate, batch size, sync) and will show up as that line on the tensorboard plot titled `data/test`.

### Using `gradient_sweep.py`

`python batch-sweep.py --max-batch-size=1000 --sweep-increment=100`

This adds a new plot for visualizing error of different batch sizes, sweeping by increments of 100 up to 1000.

# experiments

* x-axis batch size, y-axis learning rate, heatmap based on accuracy

* higher momentum 

* ensure >99% accuracy for a few different batch sizes; tweak learning rates to make it work
* make epochs constant; x-axis scales should be the same
* for the test set, just output the average over the whole test set
* first try large batches with high learning rates, cut it down until it starts working
* if after n epochs it drops, later on, try dropping learning rates later on
* target 99 on everything, lock in those learning rates 
* then generate the heatmap for just sync, and again for delayed

* LR vs. test accuracy, from 10^-6 to 10^0
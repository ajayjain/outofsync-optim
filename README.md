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

`python main.py --run-name=batch1000_sync --plot-name=different_batch_sizes --batch-size=1000 --lr=0.1 --warmup=constant --sync`

This adds a line labeled `batch1000_sync` to the plot `different_batch_sizes`, with learning rate `0.1` and warmup method `constant`.

### Using `gradient_sweep.py`

`python batch-sweep.py --max-batch-size=1000 --sweep-increment=100`

This adds a new plot for visualizing error of different batch sizes, sweeping by increments of 100 up to 1000.


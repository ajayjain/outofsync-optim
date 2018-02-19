# README

To run the basic MNIST example, do `python main.py`.

### ICLR 2018 Workshop paper & raw data

[Paper](https://www.dropbox.com/s/6kf3ogwoxo2skh2/final_paper.pdf?dl=0) submitted for the ICLR 2018 Workshop track. 

[Raw CSV Data](https://www.dropbox.com/s/skgwbqp4iwvgbn7/runs-csv.zip?dl=0). Unzip the file, and use the `Generate plots from runs.ipynb` iPython notebook to parse and visualize the data.

[TFEvents Data](https://www.dropbox.com/s/lfj7rk3dxsjvsfz/runs.zip?dl=0). Unzip the file, and run `tensorboard --logdir runs` to see training and test loss & accuracy across epochs. 

## Dependencies
* pytorch
* tensorboardX
* tensorflow

## Examples

### Using `main.py`

`python main.py --tensorboard-plot=test --log-output --batch-size=1000`

This adds a run called `MNIST_0.01_1000_0` (learning rate, batch size, sync) and will show up as that line on the tensorboard plot titled `data/test`.

### Using `batch_sweep.py`

`python batch-sweep.py --max-batch-size=1000 --sweep-increment=100`

This adds a new plot for visualizing error of different batch sizes, sweeping by increments of 100 up to 1000.

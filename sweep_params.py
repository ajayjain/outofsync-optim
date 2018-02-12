#!/usr/bin/env python3

from __future__ import print_function

import subprocess
import numpy as np

def getCmd(epochs, learning_rate, batch_size, delay, seed):
	return 'python main.py --epochs={} --lr={} --batch-size={} --delay={} --seed={} --tensorboard-plot=grid_search --log-output'.format(epochs, learning_rate, batch_size, delay, seed)


seed = 1

for delay in range(10):

	for lr_exponent in np.linspace(-5, 2, num=15):
		lr = 10 ** lr_exponent

		for batch_exponent in range(6, 12):
			bs = 2 ** batch_exponent

			cmd = getCmd(epochs=30, learning_rate=lr, batch_size=bs, delay=delay, seed=seed)

			print(cmd)
			
			subprocess.call(cmd, shell=True)

			seed += 1

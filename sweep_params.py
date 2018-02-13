#!/usr/bin/env python3

from __future__ import print_function

import subprocess
import numpy as np

def getCmd(epochs, learning_rate, batch_size, delay, seed):
	return 'python main.py --epochs={} --lr={} --batch-size={} --delay={} --seed={} --adam --tensorboard-plot=grid_search_adam --log-output'.format(epochs, learning_rate, batch_size, delay, seed)


seed = 1012010

for delay in [4, 6, 8]:
	for lr_exponent in np.linspace(-5, -.5, num=10, endpoint=False):
		lr = 10 ** lr_exponent

		for batch_exponent in range(6, 13):
			bs = 2 ** batch_exponent

			cmd = getCmd(epochs=20, learning_rate=lr, batch_size=bs, delay=delay, seed=seed)
			
			print(cmd)

			subprocess.call(cmd, shell=True)

			seed += 1

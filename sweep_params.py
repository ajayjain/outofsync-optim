#!/usr/bin/env python3

from __future__ import print_function

import subprocess
import numpy as np


def getCmd(epochs, learning_rate, batch_size, delay, seed, momentum, optimizer):
	return 'python main.py --epochs={} --lr={} --batch-size={} --delay={} --seed={} --momentum={} --optimizer={} --tensorboard-plot=grid_search --log-output'.format(epochs, learning_rate, batch_size, delay, seed, momentum, optimizer)


# seed = 1
# seed = 500
seed = 10000

class ProcessQueue():
	def __init__(self, num_procs):
		self.run_queue = []
		self.wait_queue = []
		self.num_procs = num_procs

	def queue(self, cmd):
		self.run_queue.append(cmd)

	def _run_next(self):
		if len(self.run_queue) > 0:
			cmd = self.run_queue.pop(0)
			p = subprocess.Popen(cmd, shell=True)
			self.wait_queue.append(p)

	def wait(self):
		while len(self.wait_queue) > 0 or len(self.run_queue) > 0:
			if len(self.wait_queue) < self.num_procs:
				self._run_next()

			for i, p in enumerate(self.wait_queue):
				if p.poll() is not None:
					del self.wait_queue[i]
					self._run_next()

pq = ProcessQueue(10)

for delay in [0, 1, 2, 4, 6, 8]:
	for lr_exponent in range(-12, -5):
		lr = 2 ** lr_exponent 

		for batch_exponent in range(6, 11):
			bs = 2 ** batch_exponent

			cmd = getCmd(epochs=30, learning_rate=lr, batch_size=bs, delay=delay, seed=seed, momentum=0.9, optimizer="sgd")

			print(cmd)

			pq.queue(cmd)

			seed += 1

pq.wait()

#!/usr/bin/env python3

from __future__ import print_function

import subprocess
import numpy as np
import time

from Run import Run


seed = 1
epochs = 90
delays = [0, 1]
# learning_rates = [2 ** exp for exp in range(-13, -5)]
learning_rates = [2 ** exp for exp in range(-5, -2)]
batch_sizes = [2 ** exp for exp in range(6, 11)]
momenta = [0.9, np.sqrt(0.9)]


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

			time.sleep(0.5)

pq = ProcessQueue(10)


for delay in delays:
	for mom in momenta: 
		for lr in learning_rates:
			for bs in batch_sizes:

				run = Run(
					epochs = epochs,
					learning_rate = lr,
					batch_size = bs,
					delay = delay,
					seed = seed,
					momentum = mom,
					optimizer = 'sgd'
					)

				cmd = run.to_shell()

				print(cmd)

				pq.queue(cmd)

				seed += 1

pq.wait()

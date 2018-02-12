#!/usr/bin/env python3

from __future__ import print_function

import argparse
import subprocess

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='batch sweep script')
parser.add_argument('--max-batch-size', type=int, default=1000,
                    help='max batch size for sweep (default: 1000)')
parser.add_argument('--sweep-increment', type=int, default=100,
                    help='batch sweep increment (default: 100)')

args = parser.parse_args()

sweep_increment = max(args.sweep_increment, 100)
start_size = 100
end_size = max(args.max_batch_size, 100)

def getCmd(batch_size, is_sync):
	if is_sync:
		return 'python main.py --epochs=10 --run-name=batch' + str(batch_size) + '-sync --batch-size=' + str(batch_size) + ' --sync'
	else:
		return 'python main.py --epochs=10 --run-name=batch' + str(batch_size) + ' --batch-size=' + str(batch_size)

for incr in range(int(end_size / sweep_increment)):
	batch_size = incr * sweep_increment + start_size

	subprocess.call(getCmd(batch_size, True), shell=True)
	subprocess.call(getCmd(batch_size, False), shell=True)
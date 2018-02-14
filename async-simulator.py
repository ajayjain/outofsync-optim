import numpy as np

workers = 10
batches = 100

mean_arr_time = 1
std_dev = 0.2

def get_job_time(): 
	return np.random.normal(loc=mean_arr_time, scale=std_dev)

def get_worker_timeline(id):
	deltas = [get_job_time() for _ in range(100)]
	return [(id, end_time) for end_time in np.cumsum(deltas)]

timelines = [get_worker_timeline(i) for i in range(workers)]

timeline = [evt for timeline in timelines for evt in timeline]

timeline = sorted(timeline, key=lambda evt: evt[1])

timeline = timeline[:batches]

# print(timeline)

indices = [[] for _ in range(workers)]

for (idx, (i, t)) in timeline:
	indices[i].append(idx)

delays = [np.diff(index_set) for index_set in indices]

print("delays for each worker:")
print(delays)

avg_delays = [np.mean(delay_set) for delay_set in delays]

print("avg delays for each worker:")
print(avg_delays)

avg_delay = np.mean(avg_delays)

print("avg delay statistics:")
print(avg_delay)
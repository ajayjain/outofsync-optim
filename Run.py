class Run():
	def __init__(self, 
		task='MNIST',
		epochs=90, 
		learning_rate=0.01, 
		batch_size=64, 
		delay=0, 
		seed=1, 
		momentum=0.9, 
		optimizer='sgd',
		warmup='none'):

		self.task = task
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.delay = delay
		self.seed = seed
		self.momentum = momentum
		self.optimizer = optimizer
		self.warmup = warmup


	def to_shell(self): 
		cmd = 'python main.py --epochs={} --lr={} --batch-size={} --delay={} --seed={} --momentum={} --optimizer={} --warmup={} --tensorboard-plot=grid_search --log-output'
		return cmd.format(
			self.epochs, 
			self.learning_rate, 
			self.batch_size, 
			self.delay, 
			self.seed, 
			self.momentum, 
			self.optimizer,
			self.warmup)

	def to_filename(self):
		return '_'.join([
			self.task, 
			str(self.learning_rate), 
			str(self.batch_size),
			str(self.delay),
			self.warmup,
			str(self.momentum),
			self.optimizer
			])

	@staticmethod
	def from_filename(fname):
		task, lr, batch_size, delay, warmup, momentum, optimizer = fname.split("_")
		lr = float(lr)
		batch_size = int(batch_size)
		momentum = float(momentum)
		return Run(
			task = task,
			learning_rate=lr,
			batch_size=batch_size,
			delay=delay,
			momentum=momentum,
			optimizer=optimizer,
			warmup=warmup)
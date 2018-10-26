import numpy as np

from dqn_flyweight import DQNFlyweight

class ReinforcementLearning(DQNFlyweight):
	def __init__(self, *args, **kwargs):
		super(DQNFlyweight, self).__init__()

		self.dqn = None

		if len(args) > 0:
			self.dqn = args[0]
		if 'dqn' in kwargs:
			self.dqn = kwargs['dqn']

	def steps_action(self, _act, n=4, double=False, factor=False):
		dqn, act = self.dqn, _act
		steps = dqn.step(act)
		if double:
			for i in range(n):
				steps = steps + dqn.step(act)
		elif factor:
			return ((steps for _ in np.arange(n)) for _ in np.arange(n))
		return steps

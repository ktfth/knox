from dqn_adapter import DQNAdapter

class DQNFlyweight(DQNAdapter):
	def __init__(self, *args, **kwargs):
		super(DQNAdapter, self).__init__()

		self.agent = None

		if len(args) > 0:
			self.agent = args[0]
		if 'agent' in kwargs:
			self.agent = kwargs['agent']

	def step(self, _action):
		return self.agent.step(_action)

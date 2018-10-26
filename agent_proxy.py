from reinforcement_learning import ReinforcementLearning

class AgentProxy(ReinforcementLearning):
	def __init__(self, *args, **kwargs):
		super(AgentProxy, self).__init__()

		if len(args) > 0:
			self.environment = args[0]
		if 'environment' in kwargs:
			self.environment = kwargs['environment']
		if len(args) > 1:
			self.state_size = args[1]
		if 'state_size' in kwargs:
			self.state_size = kwargs['state_size']

	def action_space_down_sample(self, s):
		method = 'action_space'
		return self.environment_fn(method).sample()

	def environment_fn(self, *ars, **kws):
		attr = ars[0]
		if 'action_space' in kws:
			attr = kws['action_space']
		if attr:
			return self.environment.__getattribute__(attr)
		return self.environment

	def step(self, *ars, **kws):
		action = ars[0]
		if 'action' in kws:
			action = kws['action']
		return self.environment.step(action)

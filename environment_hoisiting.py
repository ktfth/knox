from dqn_flyweight import DQNFlyweight

from reinforcement_learning import ReinforcementLearning

from reinforcement_learning_memento import ReinforcementLearningMemento

from agent_proxy import AgentProxy

class EnvironmentHoisiting(ReinforcementLearningMemento):
	def __init__(self, *args, **kwargs):
		super(EnvironmentHoisiting, self).__init__()

		self.name = args[0]
		if 'name' in kwargs:
			self.name = kwargs['name']

		self.make_fn = args[1]
		if 'make_fn' in kwargs:
			self.make_fn = kwargs['make_fn']

	def instance(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		instantiation = EnvironmentHoisiting(self.name, self.make)
		environment_settled = self.make(self.name)
		agent = self.agent(environment_settled, self.state_size)
		dqn = self.dqn(agent)
		net = self.net(dqn)
		return (instantiation, environment_settled,
				agent, dqn, net)

	def make(self, name):
		return self.make_fn(name)

	def agent(self, _vm, state_size):
		return AgentProxy(_vm, state_size)

	def dqn(self, _agent):
		return DQNFlyweight(agent=_agent)

	def net(self, _dqn):
		return ReinforcementLearning(_dqn)

from dqn_flyweight import DQNFlyweight

from reinforcement_learning import ReinforcementLearning

from reinforcement_learning_memento import ReinforcementLearningMemento

from agent_proxy import AgentProxy

class EnvironmentProxy(ReinforcementLearningMemento):
	def __init__(self, *args, **kwargs):
		super(EnvironmentProxy, self).__init__()

		self.instance = args[0]
		if 'name' in kwargs:
			self.instance = kwargs['instance']

	def instance(self):
		instantiation = EnvironmentProxy(self.instance)
		environment_settled = self.make(self.insntance)
		agent = self.agent(environment_settled, self.state_size)
		dqn = self.dqn(agent)
		net = self.net(dqn)
		return (environment_settled,
				agent, dqn, net)

	def make(self):
		return self.instance

	def agent(self, _vm, state_size):
		return AgentProxy(_vm, state_size)

	def dqn(self, _agent):
		return DQNFlyweight(agent=_agent)

	def net(self, _dqn):
		return ReinforcementLearning(_dqn)

import gym as g
import retro as r
import random as radix
import tensorflow as hagnar

import warnings as ignite ; ignite.simplefilter('ignore')

class DQNAdapter(object):
	def __init__(self, *args, **kwargs):
		super(type(object)).__init__()

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

class PolicyGradientBuilder(hagnar.keras.Model):
	pass

class ReinforcementLearning(DQNFlyweight):
	def __init__(self, *args, **kwargs):
		super(DQNFlyweight, self).__init__()

		self.dqn = None

		if len(args) > 0:
			self.dqn = args[0]
		if 'dqn' in kwargs:
			self.dqn = kwargs['dqn']

	def steps_action(self, _act):
		dqn = self.dqn
		act = _act
		return (dqn.step(act), dqn.step(act), dqn.step(act), dqn.step(act)), \
	           (dqn.step(act), dqn.step(act), dqn.step(act), dqn.step(act)), \
	           (dqn.step(act), dqn.step(act), dqn.step(act), dqn.step(act)), \
	           (dqn.step(act), dqn.step(act), dqn.step(act), dqn.step(act))

class AgentProxy(ReinforcementLearning):
	def __init__(self, *args, **kwargs):
		super(AgentProxy, self).__init__()

		self.environment = args[0]
		if 'environment' in kwargs:
			self.environment = kwargs['environment']

	def action_space_down_sample(self):
		return self.environment_fn('action_space').sample()

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

class EnvironmentHoisiting(object):
	def __init__(self, *args, **kwargs):
		super(EnvironmentHoisiting, self).__init__()

		self.name = args[0]
		if 'name' in kwargs:
			self.name = kwargs['name']

		self.make_fn = args[1]
		if 'make_fn' in kwargs:
			self.make_fn = kwargs['make_fn']

	def instance(self):
		instantiation = EnvironmentHoisiting(self.name, self.make)
		environment_settled = self.make(self.name)
		agent = self.agent(environment_settled)
		dqn = self.dqn(agent)
		net = self.net(dqn)
		return (instantiation, environment_settled,
				agent, dqn,
				net)

	def make(self, name):
		return self.make_fn(name)

	def agent(self, _vm):
		return AgentProxy(_vm)

	def dqn(self, _agent):
		return DQNFlyweight(agent=_agent)

	def net(self, _dqn):
		return ReinforcementLearning(_dqn)

def main(unusued_argv):
	virtualization, vm, rl, dqn, net = EnvironmentHoisiting('Enduro-ram-v0', g.make).instance()
	for e in range(10):
	    s = vm.reset()
	    for t in range(1000):
	        vm.render(mode='rgb')
	        act = rl.action_space_down_sample()
	        obs, rew, don, inf = net.steps_action(act)
	    if don:
	    	break
	vm.close()

if __name__ == '__main__':
	hagnar.logging.set_verbosity(hagnar.logging.INFO)
	hagnar.app.run(main)
import argparse
import gym as g
import retro as r
import numpy as np
import random as radix
import tensorflow as hagnar

import warnings as ignite ; ignite.simplefilter('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='rgb',
							  help='Rendering mode')
parser.add_argument('--environment', type=str, default='MsPacman-v0',
									 help='Environment name')
parser.add_argument('--episodes', type=int, default=10,
								  help='Seens episode')
parser.add_argument('--timesteps', type=int, default=1000,
								   help='Watchout series')

parser.add_argument('--policy_construct_file_path', type=str, default='solid_buma.h5',
													help='Constructing buma')
parser.add_argument('--policy_builder_file_path', type=str, default='solid_buma.h5',
												  help='Builder buma')

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

class PolicyGradientComposite(hagnar.keras.Model):
	pass

class PolicyGradientBuilder(object):
	def __init__(self, *args, **kwargs):
		super(type(object)).__init__()

	def load(self, *ars, **kws):
		pass

	def generate(self, *ars, **kws):
		return ('sample' in kws and kws['sample']) or ars[0]

	def learn(self, *ars, **kws):
		return ('samples_act' in kws and kws['samples_act']) or ars[0]

	def save(self, *ars, **kws):
		pass

class ReinforcementLearning(DQNFlyweight):
	def __init__(self, *args, **kwargs):
		super(DQNFlyweight, self).__init__()

		self.dqn = None

		if len(args) > 0:
			self.dqn = args[0]
		if 'dqn' in kwargs:
			self.dqn = kwargs['dqn']

	def steps_action(self, _act, n=4):
		dqn = self.dqn
		act = _act
		return (dqn.step(act) for _ in np.arange(n)), \
	           (dqn.step(act) for _ in np.arange(n)), \
	           (dqn.step(act) for _ in np.arange(n)), \
	           (dqn.step(act) for _ in np.arange(n))

class ReinforcementLearningMemento(object):
	pass

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

class EnvironmentHoisiting(ReinforcementLearningMemento):
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

def main(argv):
	args = parser.parse_args(argv[1:])

	virtualization, vm, rl, dqn, net = EnvironmentHoisiting(args.environment, g.make).instance()

	policy_gradient = PolicyGradientBuilder()

	policy_gradient.load(args.policy_construct_file_path)

	for e in np.arange(args.episodes):
	    s = vm.reset()
	    for t in np.arange(args.timesteps):
	        vm.render(mode=args.mode)
	        act = policy_gradient.generate(rl.action_space_down_sample())
	        obs, rew, don, inf = policy_gradient.learn(net.steps_action(act))
	        policy_gradient.save(args.policy_builder_file_path)
	    if don:
	    	break
	vm.close()

if __name__ == '__main__':
	hagnar.logging.set_verbosity(hagnar.logging.INFO)
	hagnar.app.run(main)
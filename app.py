import os
import gym as g
import threading
import numpy as np
import random as radix
import multiprocessing
import tensorflow as tf

import warnings as ignite ; ignite.simplefilter('ignore')

K = tf.keras.backend

from app_parser import parser

from dqn_flyweight import DQNFlyweight

from policy_gradient_builder import PolicyGradientBuilder

from reinforcement_learning import ReinforcementLearning

class ReinforcementLearningMemento(object):
	pass

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

def main(argv):
	args = parser.parse_args(argv[1:])

	if args.usage == 'help':
		return parser.print_help()

	state_size = args.state_size
	action_size = args.action_size

	virtualization, vm, rl, dqn, net = EnvironmentHoisiting(args.environment, g.make).instance(state_size, action_size)

	if args.state_size_environment == 'space' and vm.observation_space.shape:
		state_size = vm.observation_space.shape[0]
	if args.state_size_environment == 'space' and 'n' in dir(vm.action_space):
		action_size = vm.action_space.n
	if args.env == 'list_data' and args.env_presence != 'env_spec':
		return '\n'.join([str(name) for name in g.envs.registry.all() if str(name).find(args.env_presence) > -1])
	if args.env == 'list_data' and args.env_presence == 'env_spec':
		return '\n'.join([str(name) for name in g.envs.registry.all()])

	if args.dqn == 'haxlem':
		policy_gradient = PolicyGradientBuilder(state_size, action_size, True)
	if args.dqn == 'type':
		policy_gradient = PolicyGradientBuilder(state_size, action_size, False)

	if not 'weights' in os.listdir(os.path.join(os.getcwd())):
		os.mkdir(os.path.join(os.getcwd(), 'weights'))

	pgc_file_path = os.path.join(os.getcwd(), 'weights/%s' % args.policy_construct_file_path)

	policy_gradient.load(pgc_file_path)

	def _reinforce():
		for e in np.arange(args.episodes):
			s = vm.reset()
			if not np.asarray(s).size == 1:
				s = np.reshape(s, [1, state_size])
			for t in np.arange(args.timesteps):
				if not args.mode == 'render':
					vm.render(mode=args.mode)
				if args.mode == 'render':
					vm.render()
				act = policy_gradient.generate(s)
				act = rl.action_space_down_sample(act)
				act = net.steps_action(act)
				obs, rew, don, inf = policy_gradient.learn(act)
				# rew = rew if not don else -10
				obs = np.reshape(obs, [1, state_size])
				policy_gradient.replay(args.batch_size, args.epochs)
				policy_gradient.save(pgc_file_path)
				if don:
					s = vm.reset()
					if not np.asarray(s).size == 1:
						s = np.reshape(s, [1, state_size])
					break

	def _reinforce_cycle():
		for r in np.arange(args.reinforce):
			try:
				_reinforce()
			except MemoryError as me:
				tf.logging.debug(me)
			finally:
				break
		return policy_gradient

	trx = threading.Thread(target=_reinforce_cycle, args=())
	trx.daemon = True
	if args.daemonize == 'dqn':
		trx.daemon = False
	trx.start()

	vm.close()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)

import os
import argparse
import gym as g
import retro as r
import numpy as np
import random as radix
import tensorflow as tf

from collections import deque

import warnings as ignite ; ignite.simplefilter('ignore')

K = tf.keras.backend

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

parser.add_argument('--state_size', type=int, default=100800,
									help='Stateless definition of space based on observation')
parser.add_argument('--action_size', type=int, default=0,
									 help='Action condition')

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

class PolicyGradientComposite(tf.keras.models.Model):
	pass

class policy_gradient_h_params:
	learning_rate = 10e-9
	decay = 10e-4

class memory:
	alloc = deque(maxlen=2046)

class HuberLoss:
    def __init__(self, target, prediction):
        self.target = target
        self.prediction = prediction

    def produce_error(self):
        return self.prediction - self.target

    def square_error(self):
        return K.square(self.produce_error())

    def add_square_error(self, minima=1):
        return minima + self.square_error()

    def sqrt_error(self):
        return K.sqrt(self.add_square_error())

    def negative_sqrt_error(self, minima=1):
        return self.sqrt_error() - minima

    def mean_sqrt_error(self, axis_val=1):
        return K.mean(self.negative_sqrt_error(), axis=-axis_val)

    def eval_error(self):
        return self.mean_sqrt_error()

class QMeaning:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def max_labels_predictions(self, axis_val=1):
        return K.max(self.y_pred, axis=-axis_val)

    def mean_predictible_labels(self):
        return K.mean(self.max_labels_predictions())

    def eval_discrete(self):
        return self.mean_predictible_labels()

class PolicyGradientBuilder(object):
	def __init__(self, *args, **kwargs):
		super(type(object)).__init__()

		self.state_size = args[0]
		if 'state_size' in kwargs:
			self.state_size = kwargs['state_size']
		self.action_size = args[1]
		if 'action_size' in kwargs:
			self.action_size = kwargs['action_size']

		self.memory = memory.alloc

		self.learning_rate = policy_gradient_h_params.learning_rate
		self.decay = policy_gradient_h_params.decay

		self.model = self._compositional_meaning(self.state_size, self.action_size)
		self.target_model = self._compositional_meaning(self.state_size, self.action_size)
		self.target_model = self._compile_target(self.target_model)

		self._exchanging_rates()

	def _compositional_meaning(self, state_size, action_size):
		learning_rate = self.learning_rate
		huber_loss = self._huber_loss
		decay = self.decay
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Dense(16, input_dim=state_size))
		model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(action_size, activation=tf.keras.activations.linear))
		model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=learning_rate,
															 epsilon=K.epsilon(),
															 decay=decay),
		              loss=huber_loss,
					  metrics=[tf.keras.metrics.categorical_accuracy])
		return model

	def _compile_target(self, model):
		q_mean = self._mean_q
		learning_rate = self.learning_rate
		decay = self.decay
		model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=learning_rate,
															 epsilon=K.epsilon(),
														     decay=decay),
					  loss=q_mean)
		return model

	def _exchanging_rates(self):
		self.target_model.set_weights(self.model.get_weights())

	def _produce_rank(self, *ars, **kws):
		return tuple([v for (k, v) in kws.items()] or ars)

	def memoization(self, *ars, **kws):
		rank = self._produce_rank(*ars, **kws)
		self.memory.append(rank)
		return ars[0]

	def actual(self, state):
		if K.epsilon() >= np.random.rand():
			return np.random.randint(self.action_size)
		predicted_state = self.model.predict(state)
		if not predicted_state:
			predicted_seq = state
			return predicted_seq
		predicted_seq = np.argmax(predicted_state[0])
		return predicted_seq

	def replay(self, batch_size):
		es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
											  patience=0, verbose=0,
											  mode='auto')
		rpg = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
												   patience=5, min_lr=0.001)
		mini_batching_size = radix.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in mini_batching_size:
			target = self.model.predict(state)
			reward = reward * .022
			if done:
				target[0][action] = reward
			else:
				a = self.model.predict(next_state)[0]
				t = self.target_model.predict(next_state)[0]
				target[0][action] = reward + self.gamma * t[np.argmax(a)]
			self.model.fit(state, target,
						   epochs=10, verbose=0,
						   callbacks=[es, rpg])

	def _mean_q(self, y_true, y_pred):
		return QMeaning(y_true, y_pred).eval_discrete()

	def _huber_loss(self, target, prediction):
		return HuberLoss(target, prediction).eval_error()

	def load(self, *ars, **kws):
		filename = ars[0]
		if 'filename' in kws:
			filename = kws['filename']
		if os._exists(filename):
			self.model.load_weights(filename)
		return self.model

	def generate(self, *ars, **kws):
		return self.actual(('sample' in kws and kws['sample']) or ars[0])

	def learn(self, *ars, **kws):
		samples = ars[0]
		if 'samples' in kws:
			samples = kws['samples']
		return self.memoization(samples)

	def save(self, *ars, **kws):
		filename = ars[0]
		if 'filename' in kws:
			filename = kws['filename']
		self.model.save_weights(filename)
		return self.model

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
		return dqn.step(act)
		# return ((dqn.step(act) for _ in np.arange(n)) for _ in np.arange(n))

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
		# import pdb ; pdb.set_trace()
		# return s
		method = 'action_space'
		# state_size = self.state_size
		# action_space_sample = self.environment_fn(method).sample()
		# return np.array([[action_space_sample for x in np.arange(state_size)]])
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

	state_size = args.state_size
	action_size = args.action_size
	
	virtualization, vm, rl, dqn, net = EnvironmentHoisiting(args.environment, g.make).instance(state_size, action_size)

	# if len(vm.observation_space.shape) > 0:
	# 	state_size = vm.observation_space.shape[0]
	# if len(vm.action_space.shape) > 0 and 'n' in dir(vm.action_space):
	# 	action_size = vm.action_size.n
	# if len(vm.action_space.shape) > 0 and not 'n' in dir(vm.action_space):
	# 	action_size = vm.action_space.shape[0]

	policy_gradient = PolicyGradientBuilder(state_size, action_size)

	policy_gradient.load(args.policy_construct_file_path)

	for e in np.arange(args.episodes):
		s = vm.reset()
		s = np.reshape(s, [1, state_size])
		for t in np.arange(args.timesteps):
			if not args.mode == 'render':
				vm.render(mode=args.mode)
			if args.mode == 'render':
				vm.render()
			# act = policy_gradient.generate(rl.action_space_down_sample(s))
			act = rl.action_space_down_sample(s)
			# obs, rew, don, inf = policy_gradient.learn(net.steps_action(act))
			obs, rew, don, inf = net.steps_action(act)
			obs = np.reshape(obs, [1, state_size])
			# policy_gradient.replay(64)
			policy_gradient.save(args.policy_builder_file_path)
		if don:
			break
		# policy_gradient.replay(16)
		policy_gradient.save(args.policy_builder_file_path)
		vm.close()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
import argparse
import gym as g
import retro as r
import numpy as np
import random as radix
import tensorflow as tf

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

		self.learning_rate = policy_gradient_h_params.learning_rate
		self.decay = policy_gradient_h_params.decay

		self.model = tf.keras.models.Sequential()
		self.model = self._compositional_meaning(self.state_size, self.action_size, self.model)

		self.target_model = tf.keras.models.Sequential()
		self.target_model = self._compositional_meaning(self.state_size, self.action_size, self.target_model)
		self.target_model = self._compile_target(self.target_model)

	def _compositional_meaning(self, state_size, action_size, model):
		learning_rate = self.learning_rate
		huber_loss = self._huber_loss
		model.add(tf.keras.layers.Dense(16, input_dim=state_size))
		model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
		model.add(tf.keras.layers.Dense(action_size, activation=tf.keras.activations.linear))
		model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
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

	def _mean_q(self, y_true, y_pred):
		return QMeaning(y_true, y_pred).eval_discrete()

	def _huber_loss(self, target, prediction):
		return HuberLoss(target, prediction).eval_error()

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
		return ((dqn.step(act) for _ in np.arange(n)) for _ in np.arange(n))

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
				agent, dqn, net)

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

	state_size = args.state_size
	if len(vm.observation_space.shape) > 0:
		state_size = vm.observation_space.shape[0]
	action_size = args.action_size
	if len(vm.action_space.shape) > 0:
		action_size = vm.action_space.shape[0]

	policy_gradient = PolicyGradientBuilder(state_size, action_size)

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
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)
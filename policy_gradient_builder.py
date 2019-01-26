import os

import numpy as np

import tensorflow as tf

from policy_gradient_composite import PolicyGradientComposite

from parameters import policy_gradient_h_params

from parameters import memory

from huber_loss import HuberLoss

from qmeaning import QMeaning

K = tf.keras.backend

class PolicyGradientBuilder(object):
	def __init__(self, *args, **kwargs):
		super(type(object)).__init__()

		self.state_size = args[0]
		if 'state_size' in kwargs:
			self.state_size = kwargs['state_size']
		self.action_size = args[1]
		if 'action_size' in kwargs:
			self.action_size = kwargs['action_size']
		self.haxlem = args[2]
		if 'haxlem' in kwargs:
			self.haxlem = kwargs['haxlem']

		self.memory = memory.alloc

		self.learning_rate = policy_gradient_h_params.learning_rate
		self.epsilon = policy_gradient_h_params.epsilon
		# self.decay = policy_gradient_h_params.decay

		self.model = self._compositional_meaning(self.state_size, self.action_size, self.haxlem)
		self.target_model = self._compositional_meaning(self.state_size, self.action_size, self.haxlem)
		self.target_model = self._compile_target(self.target_model)

		self._exchanging_rates()

	def _compositional_q_meaning(self, state_size, action_size):
		learning_rate = self.learning_rate
		epsilon = self.epsilon
		huber_loss = self._huber_loss
		# decay = self.decay

		# K.set_epsilon(epsilon)

		image_input = tf.keras.layers.Input(shape=state_size)
		output_resolution = tf.keras.layers.Conv2D(filters=32, kernel_size=8,
												   strides=(4, 4), padding='valid',
												   use_bias=True, activation='relu')(image_input)
		output_resolution = tf.keras.layers.Dense(64)(output_resolution)
		output_resolution = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
												   strides=(2, 2), padding='valid',
												   activation='relu')(output_resolution)
		output_resolution = tf.keras.layers.Dense(32)(output_resolution)
		output_resolution = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(output_resolution)
		output_resolution = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
												   strides=(1, 1), padding='valid',
												   activation='relu')(output_resolution)
		output_resolution = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), padding='valid')(output_resolution)
		output_resolution = tf.keras.layers.Flatten()(output_resolution)
		output_resolution = tf.keras.layers.Dense(512, activation='relu')(output_resolution)
		output_resolution = tf.keras.layers.Dense(action_size)(output_resolution)

		return image_input, output_resolution

	def _compositional_q_meaning_model(self, state_size, action_size, haxlem=False):
		inputs, outputs = self._compositional_q_meaning(state_size, action_size)
		return tf.keras.models.Model(inputs, outputs)

	def _compositional_meaning(self, state_size, action_size, haxlem=True):
		learning_rate = self.learning_rate
		epsilon = self.epsilon
		huber_loss = self._huber_loss
		K.set_epsilon(epsilon)

		if haxlem:
			model = PolicyGradientComposite([
				tf.keras.layers.Dense(16, input_dim=state_size),
				tf.keras.layers.Dense(32, activation=tf.nn.relu),
				tf.keras.layers.Dense(32, activation=tf.nn.relu),
				tf.keras.layers.Dense(16, activation=tf.nn.relu),
				tf.keras.layers.Dense(action_size, activation=tf.keras.activations.linear),
			])
		elif not haxlem:
			model = self._compositional_q_meaning_model((state_size, state_size, state_size), action_size)

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate,
													     epsilon=K.epsilon()),
		              loss=huber_loss,
					  metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy])
		return model

	def _compile_target(self, model):
		q_mean = self._mean_q
		learning_rate = self.learning_rate
		epsilon = self.epsilon
		K.set_epsilon(epsilon)
		model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate,
														 epsilon=K.epsilon()),
					  loss=q_mean)
		return model

	def _exchanging_rates(self):
		self.target_model.set_weights(self.model.get_weights())

	def _produce_rank(self, *ars, **kws):
		return tuple([v for (k, v) in kws.items()] or ars)

	def memoization(self, *ars, **kws):
		rank = self._produce_rank(*ars, **kws)
		self.memory.append(rank)
		return rank

	def actual(self, state):
		if np.random.rand() <= K.epsilon():
			try:
				return np.random.randint(-1, self.action_size)
			except Exception as e:
				#tf.logging.debug(e)
				return int(round(radix.random() * self.action_size))

		try:
			p = self.model.predict(np.array([[state[0] for _ in np.arange(self.state_size)]]))
			if p.tolist():
				return np.argmax(p[0])
		except Exception as e:
			#tf.logging.debug(e)
			pass
		finally:
			return state

	def replay(self, batch_size, eps=1):
		es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
											  patience=0, verbose=0,
											  mode='auto')
		rpg = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
												   patience=5, min_lr=0.001)
		try:
			mini_batching_size = radix.sample(self.memory, batch_size)
			for state, action, reward, next_state, done in mini_batching_size:
				target = self.model.predict(state)
				reward = reward * .75
				if done:
					target[0][action] = reward
				else:
					a = self.model.predict(next_state)[0]
					t = self.target_model.predict(next_state)[0]
					target[0][action] = reward + self.gamma * t[np.argmax(a)]
				self.model.fit(state, target,
							   epochs=eps,
							   verbose=0, callbacks=[es, rpg])
		except Exception as e:
			#tf.logging.debug(e)
			pass
		finally:
			return (self.model, self.target_model)

	def _mean_q(self, y_true, y_pred):
		return QMeaning(y_true, y_pred).eval_discrete()

	def _huber_loss(self, target, prediction):
		# return HuberLoss(target, prediction).eval_error()
		return tf.losses.huber_loss(
			target, prediction, reduction=tf.losses.Reduction.NONE)

	def load(self, *ars, **kws):
		filename = ars[0]
		if 'filename' in kws:
			filename = kws['filename']
		if os._exists(filename):
			self.model.load_weights(filename)
		return self.model

	def generate(self, *ars, **kws):
		return self.actual(('sample' in kws and kws['sample']) or ars)

	def learn(self, *ars, **kws):
		samples = ars[0]
		if 'samples' in kws:
			samples = kws['samples']
		memoized_samples = self.memoization(samples)
		return samples

	def save(self, *ars, **kws):
		filename = ars[0]
		if 'filename' in kws:
			filename = kws['filename']
		self.model.save_weights(filename)
		return self.model

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

from policy_gradient_builder import PolicyGradientBuilder

from environment_hoisiting import EnvironmentHoisiting

def main(argv):
	args = parser.parse_args(argv[1:])

	if args.help_usage == 'program':
		return parser.print_help()

	state_size = args.state_size
	action_size = args.action_size

	if args.state_size_environment != 'space':
		state_size = int(args.state_size_environment)

	virtualization, vm, \
	rl, dqn, net = EnvironmentHoisiting(args.environment,
	                                    g.make).instance(state_size,
										                 action_size)

	if args.state_size_environment == 'space' and vm.observation_space.shape:
		state_size = vm.observation_space.shape[0]
	if args.state_size_environment == 'space' and 'n' in dir(vm.action_space):
		action_size = vm.action_space.n
	if args.env == 'list_data' and args.env_presence != 'env_spec':
		return '\n'.join([str(name) for name in g.envs.registry.all() \
		                  if str(name).find(args.env_presence) > -1])
	if args.env == 'list_data' and args.env_presence == 'env_spec':
		return '\n'.join([str(name) for name in g.envs.registry.all()])

	if args.dqn == 'haxlem':
		policy_gradient = PolicyGradientBuilder(state_size, action_size, True)
	if args.dqn == 'type':
		policy_gradient = PolicyGradientBuilder(state_size, action_size, False)

	if not 'weights' in os.listdir(os.path.join(os.getcwd())):
		os.mkdir(os.path.join(os.getcwd(), 'weights'))

	pgc_file_path = os.path.join(os.getcwd(), '%s' % args.policy_construct_file_path)

	if os._exists(pgc_file_path):
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
				try:
					act = policy_gradient.generate(s)
				except Exception as e:
					#tf.logging.debug(e)
					pass
				finally:
					act = rl.action_space_down_sample(act)
					act = net.steps_action(act)
					obs, rew, don, inf = policy_gradient.learn(act)
					rew = rew if not don else -10
					obs = np.reshape(obs, [1, state_size])
					policy_gradient.replay(args.batch_size, args.epochs)
					policy_gradient.save(pgc_file_path)
					if don:
						break
			K.clear_session()
			vm.close()

	# def _reinforce_cycle():
	# 	trx = threading.Thread(target=_reinforce, args=())
	# 	trx.daemon = True
	# 	if args.daemonize == 'dqn':
	# 		trx.daemon = False
	# 	for r in np.arange(args.reinforce):
	# 		try:
	# 			trx.start()
	# 		except MemoryError as me:
	# 			#tf.logging.debug(me)
	# 			break
	# 	return policy_gradient

	_reinforce()

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)

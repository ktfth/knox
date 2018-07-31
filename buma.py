import gym as g
import retro as r
import random as radix
import tensorflow as hagnar

class DQN(object):
	pass

class PolicyGradient(hagnar.keras.Model):
	pass

class ReinforcementLearning(DQN):
	pass

class Agent(ReinforcementLearning):
	def __init__(self, *args, **kwargs):
		super(Agent, self).__init__()

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

class Environment(object):
	def __init__(self, *args, **kwargs):
		super(Environment, self).__init__()

		self.name = args[0]
		if 'name' in kwargs:
			self.name = kwargs['name']

		self.make_fn = args[1]
		if 'make_fn' in kwargs:
			self.make_fn = kwargs['make_fn']

	def instance(self):
		instantiation = Environment(self.name, self.make)
		environment_settled = self.make(self.name)
		agent = self.agent(environment_settled)
		return (instantiation, environment_settled,
				agent)

	def make(self, name):
		return self.make_fn(name)

	def agent(self, _vm):
		return Agent(_vm)

def main(unusued_argv):
	virtualization, vm, rl = Environment('Enduro-ram-v0', g.make).instance()
	__leaf__, __trunk__, __root__ = 0, 1, -3
	for e in range(10):
	    s = vm.reset() # vm state
	    for t in range(1000):
	        vm.render(mode='rgb') # mode='rgb'
	        # env.render(mode='rgb')
	        act = rl.action_space_down_sample()
	        obs, rew, don, inf = (vm.step(act), vm.step(act), vm.step(act), vm.step(act)), \
	        					 (vm.step(act), vm.step(act), vm.step(act), vm.step(act)), \
	        					 (vm.step(act), vm.step(act), vm.step(act), vm.step(act)), \
	        					 (vm.step(act), vm.step(act), vm.step(act), vm.step(act))
	    __leaf__ *= 3 ; __trunk__ += 2 ; __root__ /= 5
	    if don:
	    	break
	    # if don[[int(round(radix.random() ** __leaf__)) for _ in range(__trunk__)][int(round(radix.random() ** __root__))]] and \
	       # rew[[int(round(radix.random() ** __root__)) for _ in range(__leaf__)] \
	          # [[int(round(radix.random() ** __trunk__)) for _ in range(__trunk__)][int(round(radix.random() ** __leaf__))]]]:
	        # break
	vm.close()

if __name__ == '__main__':
	hagnar.logging.set_verbosity(hagnar.logging.INFO)
	hagnar.app.run(main)
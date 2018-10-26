from collections import deque

class policy_gradient_h_params:
	learning_rate = 10e-9
	epsilon = 10e-7
	# decay = 10e-5

class memory:
	alloc = deque(maxlen=5000)

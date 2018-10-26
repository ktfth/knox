from collections import deque

class policy_gradient_h_params:
	learning_rate = .99
	epsilon = 10e-3
	# decay = 10e-5

class memory:
	alloc = deque(maxlen=5000)

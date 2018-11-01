import tensorflow as tf

from parameters import memory
from dqn_adapter import DQNAdapter
from dqn_flyweight import DQNFlyweight
from collections import deque
from policy_gradient_composite import PolicyGradientComposite
from parameters import policy_gradient_h_params

class AppDQNTestCase(tf.test.TestCase):
	def testDQNAdapterTypeClass(self):
		with self.test_session():
			self.assertEqual(type(DQNAdapter()),
				             DQNAdapter)

	def testDQNFlyweightTypeClass(self):
		with self.test_session():
			self.assertEqual(type(DQNFlyweight()),
						     DQNFlyweight)

class AppPolicyGradientTestCase(tf.test.TestCase):
	def testPolicyGradientCompositeTypeClass(self):
		with self.test_session():
			self.assertEqual(type(PolicyGradientComposite()),
							 PolicyGradientComposite)

class AppPolicyGradientHParamsTestCase(tf.test.TestCase):
	def testLearningRateValue(self):
		with self.test_session():
			h = policy_gradient_h_params
			expectation = h.learning_rate
			expected = 1e-08
			self.assertEqual(expectation, expected)

	def testEpsilonValue(self):
		with self.test_session():
			h = policy_gradient_h_params
			expectation = h.epsilon
			expected = 1e-06
			self.assertEqual(expectation, expected)

	# def testDecayValue(self):
	# 	with self.test_session():
	# 		h = policy_gradient_h_params
	# 		expectation = h.decay
	# 		expected = 10e-5
	# 		self.assertEqual(expectation, expected)

class AppMemoryAllocationTestCase(tf.test.TestCase):
	def testDequeType(self):
		with self.test_session():
			self.assertEqual(type(memory.alloc), deque)

if __name__ == '__main__':
	tf.test.main()

import tensorflow as tf

from app import DQNAdapter
from app import DQNFlyweight
from app import PolicyGradientComposite
from app import policy_gradient_h_params

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

class AppPolicyGradientHParams(tf.test.TestCase):
	def testLearningRateValue(self):
		with self.test_session():
			h = policy_gradient_h_params
			expectation = h.learning_rate
			expected = 10e-7
			self.assertEqual(expectation, expected)

	def testEpsilonValue(self):
		with self.test_session():
			h = policy_gradient_h_params
			expectation = h.epsilon
			expected = 10e-3
			self.assertEqual(expectation, expected)

if __name__ == '__main__':
	tf.test.main()
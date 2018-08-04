import tensorflow as tf

from app import DQNAdapter
from app import DQNFlyweight
from app import PolicyGradientComposite

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

if __name__ == '__main__':
	tf.test.main()
import tensorflow as tf

from app import DQNAdapter
from app import DQNFlyweight

class AppDQNTestCase(tf.test.TestCase):
	def testDQNAdapterTypeClass(self):
		with self.test_session():
			self.assertEqual(type(DQNAdapter()), DQNAdapter)

	def testDQNFlyweightTypeClass(self):
		with self.test_session():
			self.assertEqual(type(DQNFlyweight()), DQNFlyweight)

if __name__ == '__main__':
	tf.test.main()
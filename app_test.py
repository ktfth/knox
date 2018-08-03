import tensorflow as tf

from app import DQNAdapter

class AppTestCase(tf.test.TestCase):
	def testDQNAdapterTypeClass(self):
		with self.test_session():
			self.assertEqual(type(DQNAdapter()), DQNAdapter)

if __name__ == '__main__':
	tf.test.main()
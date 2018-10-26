import tensorflow as tf

class PolicyGradientComposite(tf.keras.models.Sequential):
	def __init__(self, *args, **kwargs):
		super(PolicyGradientComposite, self).__init__()

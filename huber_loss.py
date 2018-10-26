import tensorflow as tf

K = tf.keras.backend

class HuberLoss:
    def __init__(self, target, prediction):
        self.target = target
        self.prediction = prediction

    def produce_error(self):
        return self.prediction - self.target

    def square_error(self):
        return K.square(self.produce_error())

    def add_square_error(self, minima=1):
        return minima + self.square_error()

    def sqrt_error(self):
        return K.sqrt(self.add_square_error())

    def negative_sqrt_error(self, minima=1):
        return self.sqrt_error() - minima

    def mean_sqrt_error(self, axis_val=1):
        return K.mean(self.negative_sqrt_error(), axis=-axis_val)

    def eval_error(self):
        return self.mean_sqrt_error()

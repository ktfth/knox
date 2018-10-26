class QMeaning:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def max_labels_predictions(self, axis_val=1):
        return K.max(self.y_pred, axis=-axis_val)

    def mean_predictible_labels(self):
        return K.mean(self.max_labels_predictions())

    def eval_discrete(self):
        return self.mean_predictible_labels()

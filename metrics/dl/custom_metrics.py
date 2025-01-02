import tensorflow

class R2Score(tensorflow.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', 
                                   initializer='zeros')
        self.sst = self.add_weight(name='sst', 
                                   initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        residual = y_true - y_pred
        self.sse.assign_add(tensorflow.reduce_sum(tensorflow.square(residual)))
        self.sst.assign_add(tensorflow.reduce_sum(
            tensorflow.square(y_true-tensorflow.reduce_mean(y_true))))
    
    def result(self):
        return 1.0 - (self.sse / (self.sst+tensorflow.keras.backend.epsilon()))
    
    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)
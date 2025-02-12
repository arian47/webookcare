import tensorflow

class R2Score(tensorflow.keras.metrics.Metric):
    """
    Computes the R² (coefficient of determination) score, a common metric for regression models.
    The R² score is a measure of how well the predictions approximate the true data, with 1 being
    perfect prediction and 0 indicating that the model does not improve on simply predicting the mean.

    This implementation tracks the sum of squared errors (SSE) and the total sum of squares (SST),
    and computes the R² score as:

        R² = 1 - (SSE / SST)

    Methods
    -------
    update_state(y_true, y_pred, sample_weight=None)
        Updates the state of the metric by computing the sum of squared errors (SSE) and the total
        sum of squares (SST) based on the true values and predictions.

    result()
        Returns the current R² score, computed as 1 - (SSE / SST).

    reset_states()
        Resets the state of the metric (SSE and SST) to 0, which is useful at the start of each
        new epoch or batch.
    """
    def __init__(self, name='r2_score', **kwargs):
        """
        Initializes the R² score metric with initial values for SSE and SST.

        Parameters
        ----------
        name : str, optional
            The name of the metric. Default is 'r2_score'.

        kwargs : iterable, optional
            Additional keyword arguments for customization.
        """
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name='sse', 
                                   initializer='zeros')
        self.sst = self.add_weight(name='sst', 
                                   initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the R² score metric by computing the sum of squared errors (SSE)
        and the total sum of squares (SST) from the true and predicted values.

        Parameters
        ----------
        y_true : tensorflow.Tensor
            The true values (ground truth).

        y_pred : tensorflow.Tensor
            The predicted values by the model.

        sample_weight : tensorflow.Tensor, optional
            Weights for each sample. Default is None (no weighting).

        Notes
        -----
        This method updates the state for each batch and adds the sum of squared errors
        and total sum of squares to the existing state.
        """
        residual = y_true - y_pred
        self.sse.assign_add(tensorflow.reduce_sum(tensorflow.square(residual)))
        self.sst.assign_add(tensorflow.reduce_sum(
            tensorflow.square(y_true-tensorflow.reduce_mean(y_true))))
    
    def result(self):
        """
        Returns the current R² score.

        The R² score is computed as:

            R² = 1 - (SSE / SST)

        Returns
        -------
        tensorflow.Tensor
            The current R² score, which is a float value.
        """
        return 1.0 - (self.sse / (self.sst+tensorflow.keras.backend.epsilon()))
    
    def reset_states(self):
        """
        Resets the state of the metric (SSE and SST) to 0.

        This method is typically called at the start of each epoch or batch to clear out
        previous state values and ensure fresh calculations.

        Notes
        -----
        Useful when the metric needs to be recalculated after each epoch or batch.
        """
        self.sse.assign(0.0)
        self.sst.assign(0.0)
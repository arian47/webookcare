import tensorflow


class MLP(tensorflow.keras.Model):
    '''Multi Layer or Level ?! Perceptron
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, tvec_om:str='int', tvec_ngrams:int=None, tvec_mxlen:int=None,
                 dim=20000, dense_units:list[int]=[128,64, 1,], dropout_units:list[float]=[.3, .5],
                 dense_activations:list[str]=['relu', 'sigmoid',],
                 *args, **kwargs):
        '''initialize the densestack object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            tvec_om : str
            Default value is 'int'. specifies the output mode passed to the textvectorization layer
            possible values are 'int', 'multi_hot', 'count', 'tf_idf'.
            
            tvec_ngrams : int
            Default value is None. specifies the ngrams value passed to the textvectorization layer.
            
            dim : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            dense_units : list[int] 
            Default value is [128, 64, 1,]. specifies the number of units for the intermediary dense
            layers and the number of units for the last dense layer respectively.
            
            dense_activations : list[str]
            Default value is ['relu', 'sigmoid']. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        # self.tvec_om = tvec_om
        # self.tvec_ngrams = tvec_ngrams
        # self.tvec_mxlen = tvec_mxlen
        # self.dim = dim
        self.dense_units = dense_units
        self.dropout_units = dropout_units
        self.dense_activations = dense_activations
        # self.input = tensorflow.keras.layers.Input(shape=(1,), dtype='string') # not needed
        # self.text_vec_1 = tensorflow.keras.layers.TextVectorization(max_tokens=dim, output_mode=tvec_om,
                                                                    #  ngrams=tvec_ngrams, output_sequence_length=tvec_mxlen)
        # self.input_l = tensorflow.keras.layers.InputLayer(input_shape=(203,))
        self.dense_1 = tensorflow.keras.layers.Dense(dense_units[0], activation=dense_activations[0])
        self.dropout_1 = tensorflow.keras.layers.Dropout(dropout_units[0])
        self.dense_2 = tensorflow.keras.layers.Dense(dense_units[1], activation=dense_activations[0])
        self.dropout_2 = tensorflow.keras.layers.Dropout(dropout_units[-1])
        self.dense_3 = tensorflow.keras.layers.Dense(dense_units[-1], activation=dense_activations[-1])
        # self.loss_metric = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        # self.mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        # self.acc_metric = tensorflow.keras.metrics.Accuracy(name='accuracy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        # self.optimizer = tensorflow.keras.optimizers.RMSprop()
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        # i = self.text_vec_1(inputs)
        # i = self.input_l(inputs)
        i = self.dense_1(inputs)
        i = self.dense_2(i)
        i = self.dropout_1(i)
        i = self.dense_3(i)
        return i
    
    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        and encoder obj.
        
        Returns
        -------
        dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'tvec_om' : self.tvec_om,
                'tvec_ngrams' : self.tvec_ngrams,
                'tvec_mxlen' : self.tvec_mxlen,
                'dim' : self.dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
                # "sublayer": keras.saving.serialize_keras_object(self.sublayer)
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        # sublayer_config = config.pop("sublayer")
        # sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        # return cls(sublayer, **config)
        return cls(**config)
        
    def compile(self, 
                optimizer=tensorflow.keras.optimizers.RMSprop(), 
                loss_fn=tensorflow.keras.losses.BinaryCrossentropy()):
        super().compile()
        # self.optimizer = tensorflow.keras.optimizers.RMSprop()
        # self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        self.optimizer=optimizer
        self.bce=loss_fn
        
    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss
    
    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_, training=True) # will return batch of (NUM_UNIQUE_TERM,) shaped vectors
            loss = self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics
    
    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics

class BasicDense(tensorflow.keras.Model):
    '''BasicDense is a simple model of a dense and a dropout layer
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, tvec_om:str='int', tvec_ngrams:int=None, tvec_mxlen:int=None,
                 dim=20000, dense_units:list[int]=[16, 1,], dropout_units:float=.5,
                 dense_activations:list[str]= ['relu', 'sigmoid',],
                 *args, **kwargs):
        '''initialize the densestack object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            tvec_om : str
            Default value is 'int'. specifies the output mode passed to the textvectorization layer
            possible values are 'int', 'multi_hot', 'count', 'tf_idf'.
            
            tvec_ngrams : int
            Default value is None. specifies the ngrams value passed to the textvectorization layer.
            
            dim : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            dense_units : list[int] 
            Default value is [16, 1,]. specifies the number of units for the intermediary dense
            layer and the number of units for the last dense layer respectively.
            
            dense_activations : list[str]
            Default value is ['relu', 'sigmoid']. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.tvec_om = tvec_om
        self.tvec_ngrams = tvec_ngrams
        self.tvec_mxlen = tvec_mxlen
        self.dim = dim
        self.dense_units = dense_units
        self.dropout_units = dropout_units
        self.dense_activations = dense_activations
        # self.input = tensorflow.keras.layers.Input(shape=(1,), dtype='string') # not needed
        self.text_vec_1 = tensorflow.keras.layers.text_vectorization(max_tokens=dim, output_mode=tvec_om,
                                                                     ngrams=tvec_ngrams, output_sequence_length=tvec_mxlen)
        self.dense_1 = tensorflow.keras.layers.Dense(dense_units[0], activation=dense_activations[0])
        self.dropout_1 = tensorflow.keras.layers.Dropout(dropout_units)
        self.dense_2 = tensorflow.keras.layers.Dense(dense_units[-1], activation=dense_activations[1])
        # self.loss_metric = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        # self.mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        # self.acc_metric = tensorflow.keras.metrics.Accuracy(name='accuracy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        self.optimizer = tensorflow.keras.optimizers.RMSprop()
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        # i = self.text_vec_1(inputs)
        i = self.dense_1(inputs)
        i = self.dropout_1(i)
        i = self.dense_2(i)
        return i
    
    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        and encoder obj.
        
        Returns
        -------
        dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'tvec_om' : self.tvec_om,
                'tvec_ngrams' : self.tvec_ngrams,
                'tvec_mxlen' : self.tvec_mxlen,
                'dim' : self.dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
                # "sublayer": keras.saving.serialize_keras_object(self.sublayer)
            }
        )
        return config
    
    # @classmethod
    # def from_config(cls, config):
        # sublayer_config = config.pop("sublayer")
        # sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        # return cls(sublayer, **config)
        
    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss
    
    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_)
            loss= self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics
    
    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics


class BasicEmbedding(tensorflow.keras.Model):
    '''BasicEmbedding is a simple model of embedding, global average pooling, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, tvec_om:str='int', tvec_ngrams:int=None, tvec_mxlen:int=None, dim=20000, o_dim:int=128, dense_units:list[int]=[16, 1,],
                 dropout_units:float=.5, dense_activations:list[str]= ['relu', 'sigmoid'], mask_zero:bool=False, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            tvec_om : str
            Default value is 'int'. specifies the output mode passed to the textvectorization layer
            possible values are 'int', 'multi_hot', 'count', 'tf_idf'.
            
            tvec_ngrams : int
            Default value is None. specifies the ngrams value passed to the textvectorization layer.
            
            dims : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            o_dim : int
            Default value is 128. specifies the output dimension of the embedding layer.
            
            dense_units : list[int] 
            Default value is [16, 1,]. specifies the number of units for the intermediary dense
            layer and the number of units for the last dense layer respectively.
            
            dense_activations : list[str]
            Default value is ['relu', 'sigmoid']. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.tvec_om = tvec_om
        self.tvec_ngrams = tvec_ngrams
        self.tvec_mxlen = tvec_mxlen
        self.dim = dim
        self.o_dim = o_dim
        self.dense_units = dense_units
        self.dropout_units = dropout_units
        self.dense_activations = dense_activations
        self.mask_zero = mask_zero
        # self.input = tensorflow.keras.layers.Input(shape=(1,), dtype='string') # not needed
        self.text_vec_1 = tensorflow.keras.layers.TextVectorization(max_tokens=dim, output_mode=tvec_om,
                                                                     ngrams=tvec_ngrams, output_sequence_length=tvec_mxlen)
        self.embedding = tensorflow.keras.layers.Embedding(input_dim=dim, output_dim=o_dim, mask_zero=self.mask_zero)
        self.glob_avg_pool = tensorflow.keras.layers.GlobalAveragePooling1D()
        self.dense_1 = tensorflow.keras.layers.Dense(dense_units[0], activation=dense_activations[0])
        self.dropout_1 = tensorflow.keras.layers.Dropout(dropout_units)
        self.dense_2 = tensorflow.keras.layers.Dense(dense_units[-1], activation=dense_activations[-1])
        # self.loss_metric = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        # self.mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        # self.acc_metric = tensorflow.keras.metrics.Accuracy(name='accuracy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        self.optimizer = tensorflow.keras.optimizers.RMSprop()
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        
        
        
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        # i = self.text_vec_1(inputs)
        i = self.embedding(inputs)
        i = self.glob_avg_pool(i)
        i = self.dense_1(i)
        i = self.dropout_1(i)
        i = self.dense_2(i)
        return i
    
    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        and encoder obj.
        
        Returns
        -------
        dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'tvec_om' : self.tvec_om,
                'tvec_ngrams' : self.tvec_ngrams,
                'tvec_mxlen' : self.tvec_mxlen,
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
                # "sublayer": keras.saving.serialize_keras_object(self.sublayer)
            }
        )
        return config
    
    # @classmethod
    # def from_config(cls, config):
        # sublayer_config = config.pop("sublayer")
        # sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        # return cls(sublayer, **config)
        
    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss
    
    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        # self.mae_metric.update_state(real_data, generated_data)
        # self.acc_metric
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_)
            loss = self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics
    
    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics
    
class BasicLSTM(tensorflow.keras.Model):
    '''BasicLSTM is a simple model of embedding, bidirectional, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, tvec_om:str='int', tvec_ngrams:int=None, tvec_mxlen:int=None, dim=20000, o_dim:int=256, dense_units:int=1,
                 dropout_units:float=.5, dense_activations:str='sigmoid', mask_zero:bool=False, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            tvec_om : str
            Default value is 'int'. specifies the output mode passed to the textvectorization layer
            possible values are 'int', 'multi_hot', 'count', 'tf_idf'.
            
            tvec_ngrams : int
            Default value is None. specifies the ngrams value passed to the textvectorization layer.
            
            dims : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            o_dim : int
            Default value is 128. specifies the output dimension of the embedding layer.
            
            dense_units : int 
            Default value is 1. specifies the number of units for the intermediary dense
            layer and the number of units for the last dense layer respectively.
            
            dense_activations : str
            Default value is 'sigmoid'. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.tvec_om = tvec_om
        self.tvec_ngrams = tvec_ngrams
        self.tvec_mxlen = tvec_mxlen
        self.dim = dim
        self.o_dim = o_dim
        self.dense_units = dense_units
        self.dropout_units = dropout_units
        self.dense_activations = dense_activations
        self.mask_zero = mask_zero
        # self.input = tensorflow.keras.layers.Input(shape=(1,), dtype='string') # not needed
        self.text_vec_1 = tensorflow.keras.layers.TextVectorization(max_tokens=dim, output_mode=tvec_om,
                                                                     ngrams=tvec_ngrams, output_sequence_length=tvec_mxlen)
        self.embedding = tensorflow.keras.layers.Embedding(input_dim=dim, output_dim=o_dim,
                                                           mask_zero=self.mask_zero)
        self.bidir_lstm = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32))
        self.dense = tensorflow.keras.layers.Dense(dense_units, activation=dense_activations)
        self.dropout = tensorflow.keras.layers.Dropout(dropout_units)
        # self.loss_metric = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        # self.mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        # self.acc_metric = tensorflow.keras.metrics.Accuracy(name='accuracy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        self.optimizer = tensorflow.keras.optimizers.RMSprop()
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        
        
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        # i = self.text_vec_1(inputs)
        i = self.embedding(inputs)
        i = self.bidir_lstm(i)
        i = self.dense(i)
        i = self.dropout(i)
        return i
    
    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        and encoder obj.
        
        Returns
        -------
        dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'tvec_om' : self.tvec_om,
                'tvec_ngrams' : self.tvec_ngrams,
                'tvec_mxlen' : self.tvec_mxlen,
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
                # "sublayer": keras.saving.serialize_keras_object(self.sublayer)
            }
        )
        return config
    
    # @classmethod
    # def from_config(cls, config):
        # sublayer_config = config.pop("sublayer")
        # sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        # return cls(sublayer, **config)
        
    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss
    
    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        # self.mae_metric.update_state(real_data, generated_data)
        # self.acc_metric
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_)
            loss = self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics
    
    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics

class PositionalEmbedding(tensorflow.keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_embeddings = tensorflow.keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.position_embeddings = tensorflow.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def call(self, inputs):
        length = tensorflow.shape(inputs)[-1]
        positions = tensorflow.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tensorflow.math.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim' : self.output_dim,
            'sequence_length' : self.sequence_length,
            'input_dim' : self.input_dim,
        })
        return config



class DenseStack(tensorflow.keras.layers.Layer):
    '''DenseStack is a simple dense stack used by transformer encoder.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, dense_dim:int, embed_dim:int, activation:str=None, 
                 name:str=None,  *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            dense_dim : int
            number of units param passed to the intermediary dense layer.
            
            embed_dim : int
            number of units passed to the last dense layer.
            
            activation : str
            Default value is None. activation function supplied to the dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.name = name
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim
        self.activation = activation
        self.dense_1 = tensorflow.keras.layers.Dense(
            self.dense_dim,
            activation=self.activation
        )
        self.dense_2 = tensorflow.keras.layers.Dense(
            self.embed_dim
        )
        
        def call(self, inputs):
            '''predicts the value based on batches of input strings.
        
            Parameters
            ----------
                inputs : iterbale
                batches of strings and targets

            Returns
            -------
                i : float
                the predicted value based on weights and input value
            '''
            i = self.dense_1(inputs)
            i = self.dense_2(i)
            return i
        
        def get_config(self):
            '''builds the dictionary which hold necessary parameters to reconstruct
            an encoder obj.

            Returns
            -------
            dict obj
            '''
            config = super().get_config()
            config.update(
                {
                    'name' : self.name,
                    'dense_dim' : self.dense_dim,
                    'embed_dim' : self.embed_dim,
                    'activation' : self.activation
                }
            )
            return config

class TransformerEncoder(tensorflow.keras.layers.Layer):
    '''TransformerEncoder is a simple encoder based on transformer architectures based on
    multiheadattention, dense stacks, layernormalization layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, embed_dim, dense_dim, num_heads, name:str=None, 
                 dense_name:str=None, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            Default value is None. user specified name for the encoder layer obj.
            
            embed_dim : int
            number of units passed to the last dense layer.
            
            dense_dim : int 
            specifies the number of units for the intermediary dense
            
            num_heads : int
            num heads for the multihead layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.name = name
        self.dense_name = dense_name
        self.attention = tensorflow.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.dense_proj = DenseStack(self.dense_dim, self.embed_dim, 'relu',
                                     name=dense_name)
        self.layernorm_1 = tensorflow.keras.layers.LayerNormalization()
        self.layernorm_2 = tensorflow.keras.layers.LayerNormalization()
    
    def call(self, inputs, mask=None):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        if mask is not None:
            mask = mask[:, tensorflow.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'dense_name' : self.dense_name,
                'embed_dim' : self.embed_dim,
                'num_heads' : self.num_heads,
                'dense_dim' : self.dense_dim,
        })
        return config

    # def compute_loss(self, real_data, generated_data):
        # loss = self.bce(real_data, generated_data)
        # return loss


class Transformer(tensorflow.keras.Model):
    '''Transformer is a simple model of embedding, transformer encoder, global max pooling,
    dense, and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name:str=None, enc_name:str=None, dense_stack_name:str=None, 
                 vocab_size:int=20000, embed_dim:int=256, num_heads:int=2, 
                 dense_dim:int=32, dropout_val:float=.5, dense_unit:int=1, 
                 dense_activation:str='sigmoid', *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            enc_name : str
            user specified name for the transformer encoder layer.
            
            dense_stack_name : str
            user specified name for dense_stack layer.
            
            vocab_size : int
            Default value is 20000. specifies the input dim val passed to the embedding layer.
            
            embed_dim : int
            Default value is 256. specifies the output dim val passed to the embedding layer
            and also the number of units for the last dense layer of the transformer encoder.
            
            num_heads : int
            Default value is 2. number of attention heads passed to the attention head layer.
            
            dense_dim : int
            Default value is 32. number of units for the intermediary dense layer that transformer
            encoder uses.
            
            dropout_val : float
            Default value is 0.5. dropout value used for the dropout layer.
            
            dense_unit : int 
            Default value is 1. specifies the number of units for the last dense
            layer.
            
            dense_activation : str
            Default value is 'sigmoid'. activation function supplied to the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.name = name
        self.enc_name = enc_name
        self.dense_stack_name = dense_stack_name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.dropout_val = dropout_val
        self.dense_unit = dense_unit
        self.dense_activation = dense_activation
        self.embed_1 = tensorflow.keras.layers.Embedding(vocab_size, embed_dim,
                                                         input_shape=(None,))
        self.trf_enc = TransformerEncoder(embed_dim, dense_dim, num_heads,
                                          name=enc_name, dense_name=dense_stack_name)
        self.globmxp = tensorflow.keras.layers.GlobalMaxPooling1D()
        self.dropout = tensorflow.keras.layers.Dropout(self.dropout_val)
        self.dense = tensorflow.keras.layers.Dense(self.dense_unit, 
                                                   activation=self.dense_activation)
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        self.optimizer = tensorflow.keras.optimizers.RMSprop()
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()

    
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        i = self.embed_1(inputs)
        i = self.trf_enc(i)
        i = self.globmxp(i)
        i = self.dropout(i)
        i = self.dense(i)
        return i
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'name' : self.name,
            'enc_name' : self.enc_name,
            'dense_stack_name' : self.dense_stack_name,
            'vocab_size' : self.vocab_size,
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
            'dense_dim' : self.dense_dim,
            'dropout_val' : self.dropout_val,
            'dense_unit' : self.dense_unit,
            'dense_activation' : self.dense_activation
        })
        return config

    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss

    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_) # will return batch of (NUM_UNIQUE_TERM,) shaped vectors
            loss = self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics
    
    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics
    

class LSTM(tensorflow.keras.Model):
    '''BasicLSTM is a simple model of embedding, bidirectional, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, tvec_om:str='int', tvec_ngrams:int=None, tvec_mxlen:int=None, dim=20000, o_dim:int=256, dense_units:list[int]=[32, 1],
                 dropout_units:float=.5, dense_activations:str='sigmoid', seq_len:int=None, 
                 num_heads:int=2, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            tvec_om : str
            Default value is 'int'. specifies the output mode passed to the textvectorization layer
            possible values are 'int', 'multi_hot', 'count', 'tf_idf'.
            
            tvec_ngrams : int
            Default value is None. specifies the ngrams value passed to the textvectorization layer.
            
            dims : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            o_dim : int
            Default value is 128. specifies the output dimension of the embedding layer.
            
            dense_units : int 
            Default value is 1. specifies the number of units for the intermediary dense
            layer and the number of units for the last dense layer respectively.
            
            dropout_units : float
            Default value is 0.5. dropout value used for the dropout layer.
            
            dense_activations : str
            Default value is 'sigmoid'. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            seq_len : int 
            Default value is None. sequence length value passed to the positional embedding layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        super().__init__(name=name, *args, **kwargs)
        self.tvec_om = tvec_om
        self.tvec_ngrams = tvec_ngrams
        self.tvec_mxlen = tvec_mxlen
        self.dim = dim
        self.o_dim = o_dim
        self.dense_units = dense_units
        self.dropout_units = dropout_units
        self.dense_activations = dense_activations
        # self.mask_zero = mask_zero
        self.seq_len = seq_len
        self.num_heads = num_heads
        # self.input = tensorflow.keras.layers.Input(shape=(1,), dtype='string') # not needed
        self.text_vec_1 = tensorflow.keras.layers.TextVectorization(max_tokens=dim, output_mode=tvec_om,
                                                                    ngrams=tvec_ngrams, output_sequence_length=tvec_mxlen)
        self.positional_embed = PositionalEmbedding(self.seq_len, self.dim, self.o_dim)
        self.encoder = TransformerEncoder(self.o_dim, self.dense_units[0], num_heads)
        self.glob_max_pool = tensorflow.keras.layers.GlobalMaxPooling1D()
        # self.embedding = tensorflow.keras.layers.Embedding(input_dim=dim, output_dim=o_dim,
                                                        #    mask_zero=self.mask_zero)
        # self.bidir_lstm = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32))
        self.dense = tensorflow.keras.layers.Dense(dense_units[1], activation=dense_activations)
        self.dropout = tensorflow.keras.layers.Dropout(dropout_units)
        # self.loss_metric = tensorflow.keras.metrics.MeanSquaredError(name='mse')
        # self.mae_metric = tensorflow.keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name='binary_crossentropy')
        # self.acc_metric = tensorflow.keras.metrics.Accuracy(name='accuracy')
        self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name='accuracy')
        self.bce = tensorflow.keras.losses.BinaryCrossentropy()
        self.optimizer = tensorflow.keras.optimizers.RMSprop()
        
        
    def call(self, inputs):
        '''predicts the value based on batches of input strings.
        
        Parameters
        ----------
            inputs : iterbale
            batches of strings and targets
            
        Returns
        -------
            i : float
            the predicted value based on weights and input value
        '''
        # i = self.text_vec_1(inputs)
        i = self.positional_embed(inputs)
        i = self.encoder(i)
        i = self.glob_max_pool(i)
        # i = self.embedding(i)
        # i = self.bidir_lstm(i)
        i = self.dropout(i)
        i = self.dense(i)
        return i
    
    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        and encoder obj.
        
        Returns
        -------
        dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'name' : self.name,
                'tvec_om' : self.tvec_om,
                'tvec_ngrams' : self.tvec_ngrams,
                'tvec_mxlen' : self.tvec_mxlen,
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
                # "sublayer": keras.saving.serialize_keras_object(self.sublayer)
            }
        )
        return config
    
    # @classmethod
    # def from_config(cls, config):
        # sublayer_config = config.pop("sublayer")
        # sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        # return cls(sublayer, **config)
        
    def compute_loss(self, real_data, generated_data):
        loss = self.bce(real_data, generated_data)
        return loss
    
    def reset_metrics(self):
        self.loss_metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, generated_data)
        self.acc_metric.update_state(real_data, generated_data)
        # self.mae_metric.update_state(real_data, generated_data)
        # self.acc_metric
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_)
            loss = self.compute_loss(target, output)
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        metrics = self.compute_metrics(target, output)
        return metrics


    def test_step(self, inputs):
        '''Performs a testing step.
        
        Parameters
        ----------
        inputs : tuple
            A tuple of input data and target data.
        
        Returns
        -------
        metrics : dict
            Dictionary of metric names and their values.
        '''
        input_, target = inputs
        output = self(input_, training=False)
        loss = self.compute_loss(target, output)
        metrics = self.compute_metrics(target, output)
        return metrics
    

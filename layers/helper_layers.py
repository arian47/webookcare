import tensorflow

# class ProcessData:
#     def __init__(self, input_data, tvec_mt:int=20000, tvec_om:str='int', 
#                  ngrams:int=1, *args, **kwargs):
#         self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
#                                                               output_mode=tvec_om)
#         self.tvec.adapt(input_data)

# class VectorizePrediction:
#     def __init__(self, input_data, tvec_mt:int=20000, tvec_om:str='int', 
#                  ngrams:int=1, vocab:list[str]=None, pad_to_max_tokens:bool=False,
#                  output_sequence_length:int=None, *args, **kwargs):
#         self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
#                                                               output_mode=tvec_om,
#                                                               vocabulary=vocab,
#                                                               pad_to_max_tokens=pad_to_max_tokens,
#                                                               output_sequence_length=output_sequence_length)


class ProcessData:
    """Helper layer used for vectorizing the training or data used for prediction tasks.
    """
    def __init__(self, training:bool=True, input_data:tensorflow.Tensor=None, tvec_mt:int=20000, tvec_om:str='int', 
                 vocab:list[str]=None, pad_to_max_tokens:bool=False,
                 output_sequence_length:int=None, *args, **kwargs):
        '''initialize the vectorizer layer object.
        
        Parameters
        ----------
            training : bool
            Default value is True. used to determine whether vectorization is done for training or prediction.
            
            input_data : tensorflow.Tensor
            input data to be used for the training step if passed in.
            
            tvec_mt : int
            Default value is True. max val for vectorization tokens.
            
            tvec_om : str
            Default value is 'int'. output values type for the vectorized tensor of input data.
            
            vocab : list[str]
            Default value is None. vocabulary passed for the predictions vectorization step.
            
            pad_to_max_tokens : bool
            Default value is False. whether to pad to max tokens val or not.
            
            output_sequence_length : int
            Default value is None.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        if training:
            assert input_data, 'Training was chosen with no input data!'
            self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
                                                                  output_mode=tvec_om)
            self.tvec.adapt(input_data)
        else:
            # assert vocab, 'Vocabulary not provided for the prediction step!'
            self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
                                                                  output_mode=tvec_om,
                                                                  vocabulary=vocab,
                                                                  pad_to_max_tokens=pad_to_max_tokens,
                                                                  output_sequence_length=output_sequence_length)

# TODO: remove unnecessary layers  
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


class PositionalEmbedding(tensorflow.keras.layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_embeddings = tensorflow.keras.layers.Embedding(
            input_dim=input_dim, 
            output_dim=output_dim
        )
        self.position_embeddings = tensorflow.keras.layers.Embedding(
            input_dim=sequence_length, 
            output_dim=output_dim
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
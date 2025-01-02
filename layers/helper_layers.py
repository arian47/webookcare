import tensorflow
class ProcessData:
    """
    A helper class for vectorizing data used in training or prediction tasks.

    This class prepares the input data by converting it into a format suitable for model processing. 
    It handles the vectorization process and supports both training and prediction phases, 
    allowing for flexibility in how the data is handled.

    Parameters
    ----------
    training : bool, optional, default=True
        A flag to specify whether the vectorization is for training or prediction.
        If True, the class assumes the data is for training. If False, it assumes the data is for prediction.

    input_data : tensorflow.Tensor, optional
        The input data to be used for training. If provided, it will be processed for training purposes.
        If this is None, the vectorizer can still be used for prediction tasks.

    tvec_mt : int, optional, default=20000
        The maximum number of tokens for vectorization. This parameter determines the maximum size 
        of the tokenized data for processing.

    tvec_om : str, optional, default='int'
        The output type for the vectorized tensor. This could be a string indicating the desired 
        output data type, such as 'int' for integers, or other valid TensorFlow data types.

    vocab : list of str, optional
        A list of strings representing the vocabulary used for the prediction vectorization process. 
        If not provided, the vocabulary is inferred from the data.

    pad_to_max_tokens : bool, optional, default=False
        Whether to pad the input data to the maximum token length. If True, the input sequences are 
        padded to the maximum length, which is defined by the model or `tvec_mt`.

    output_sequence_length : int, optional, default=None
        The desired output sequence length. If specified, all sequences will be padded or truncated 
        to this length during vectorization.

    *args, **kwargs : optional
        Additional arguments or keyword arguments that can be passed to the model or processing function. 
        These could include settings for batch size, training configurations, etc.

    Attributes
    ----------
    training : bool
        Indicates whether the class is set up for training or prediction.

    input_data : tensorflow.Tensor or None
        The input data to be processed.

    tvec_mt : int
        The maximum number of tokens for vectorization.

    tvec_om : str
        The output type for the vectorized tensor.

    vocab : list of str or None
        The vocabulary used for prediction vectorization.

    pad_to_max_tokens : bool
        Whether to pad input sequences to the maximum token length.

    output_sequence_length : int or None
        The target output sequence length.

    """
    def __init__(self, training:bool=True, input_data:tensorflow.Tensor=None, tvec_mt:int=20000, tvec_om:str='int', 
                 vocab:list[str]=None, pad_to_max_tokens:bool=False,
                 output_sequence_length:int=None, *args, **kwargs):
        """
        Initialize the vectorizer layer object.

        Parameters
        ----------
        training : bool, optional, default=True
            Determines whether the vectorization is for training or prediction.

        input_data : tensorflow.Tensor, optional
            The input data to be processed for training or prediction.

        tvec_mt : int, optional, default=20000
            The maximum number of tokens for the vectorization process.

        tvec_om : str, optional, default='int'
            The output type for the vectorized tensor, such as 'int' for integers.

        vocab : list of str, optional
            Vocabulary used for the prediction vectorization process.

        pad_to_max_tokens : bool, optional, default=False
            If True, the input data will be padded to the maximum token length.

        output_sequence_length : int, optional
            The desired output sequence length for vectorized data.

        kwargs : iterable, optional
            Additional keyword arguments for custom configurations.
        """
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
    """
    DenseStack is a simple dense stack layer used in transformer encoders.
    It consists of two fully connected dense layers that perform transformations
    on input data, typically used for feature extraction or prediction.

    Methods
    -------
    call(inputs)
        Takes a batch of input data and passes it through two dense layers.
    get_config()
        Returns the configuration of the DenseStack layer for reconstruction.
    """
    def __init__(self, dense_dim:int, embed_dim:int, activation:str=None, 
                 name:str=None,  *args, **kwargs):
        """
        Initializes the DenseStack layer object.

        Parameters
        ----------
        name : str, optional
            The name for the layer instance. Default is None.

        dense_dim : int
            The number of units in the first dense layer.

        embed_dim : int
            The number of units in the second dense layer (output size).

        activation : str, optional, default=None
            The activation function for the first dense layer. If None, no activation function is applied.

        kwargs : iterable, optional
            Additional keyword arguments for further customization.
        """
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
            """
            Performs forward pass by processing input data through the two dense layers.

            Parameters
            ----------
            inputs : iterable
                A batch of input data (such as strings or numerical tensors).

            Returns
            -------
            tensorflow.Tensor
                The transformed output from the second dense layer (embedded representation).
            """
            i = self.dense_1(inputs)
            i = self.dense_2(i)
            return i
        
        def get_config(self):
            """
            Returns the configuration of the DenseStack layer for reinitialization.

            Returns
            -------
            dict
                A dictionary containing the configuration parameters of the layer.
            """
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
    """
    TransformerEncoder implements a transformer encoder layer based on multi-head attention,
    dense layers, and layer normalization. It is commonly used as part of transformer-based models
    for sequence processing tasks like natural language processing (NLP) and time-series analysis.

    Methods
    -------
    call(inputs, mask=None)
        Performs the forward pass using multi-head attention followed by dense projections and layer normalization.
    """
    def __init__(self, embed_dim, dense_dim, num_heads, name:str=None, 
                 dense_name:str=None, *args, **kwargs):
        """
        Initializes the TransformerEncoder layer object.

        Parameters
        ----------
        embed_dim : int
            The number of units in the final output of the multi-head attention layer. This defines the
            dimensionality of the embedding space.

        dense_dim : int
            The number of units in the intermediate dense layer, which is applied to the attention output.

        num_heads : int
            The number of attention heads in the multi-head attention mechanism. This controls how many
            attention processes run in parallel.

        name : str, optional
            The name of the encoder layer instance. Default is None.

        dense_name : str, optional
            The name for the dense stack layer. Default is None.

        kwargs : iterable, optional
            Additional keyword arguments for customization.
        """
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
        """
        Performs the forward pass through the encoder layer, which includes applying multi-head attention,
        adding residual connections, passing through a dense layer, and normalizing the outputs.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            The input data, typically a batch of sequences (e.g., tokenized text or numerical data).

        mask : tensorflow.Tensor, optional
            An optional attention mask to prevent attention to certain positions (e.g., padding tokens).

        Returns
        -------
        tensorflow.Tensor
            The output of the encoder after applying attention, dense projection, and normalization.
        """
        if mask is not None:
            mask = mask[:, tensorflow.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        """
        Returns the configuration of the TransformerEncoder layer, which includes the layer's hyperparameters.
        This configuration is used for reinitializing the layer with the same parameters.

        Returns
        -------
        dict
            A dictionary containing the configuration of the layer, including all the parameters
            needed for layer reconstruction. The dictionary contains:
                - 'name': The name of the layer.
                - 'dense_name': The name of the dense projection layer.
                - 'embed_dim': The embedding dimension used in the multi-head attention.
                - 'num_heads': The number of attention heads in the multi-head attention.
                - 'dense_dim': The dimension of the intermediate dense layer.
        """
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
    """
    A layer that adds positional encoding to the input token embeddings.
    This is commonly used in transformer models to incorporate positional information,
    as transformer models are not inherently sequential and do not have information 
    about the order of tokens.

    The layer computes the sum of token embeddings and positional embeddings, which 
    are then used as input to further layers.

    Parameters
    ----------
    sequence_length : int
        The length of the input sequences (i.e., the maximum number of positions).
    
    input_dim : int
        The size of the vocabulary (i.e., the number of unique tokens that can be embedded).
    
    output_dim : int
        The dimensionality of the embedding space for both token and positional embeddings.

    Methods
    -------
    call(inputs)
        Adds token and positional embeddings to the input data.

    compute_mask(inputs, mask=None)
        Computes the mask for the input sequence to handle padding tokens (zeros).
        
    get_config()
        Returns the configuration of the PositionalEmbedding layer for reinitialization.
    """
    def __init__(self, sequence_length, input_dim, output_dim, *args, **kwargs):
        """
        Initializes the PositionalEmbedding layer with token and positional embedding layers.

        Parameters
        ----------
        sequence_length : int
            The length of the input sequence, which determines the number of positions.
        
        input_dim : int
            The size of the vocabulary.
        
        output_dim : int
            The embedding dimension for both token and positional embeddings.
        
        kwargs : iterable, optional
            Additional arguments for layer customization.
        """
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
        """
        Performs the forward pass by adding token embeddings and positional embeddings.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            A tensor representing tokenized input sequences, with shape `(batch_size, sequence_length)`.
        
        Returns
        -------
        tensorflow.Tensor
            The tensor resulting from adding token embeddings and positional embeddings.
        """
        length = tensorflow.shape(inputs)[-1]
        positions = tensorflow.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        """
        Computes the mask for the input sequence to handle padding tokens (zeros).

        Parameters
        ----------
        inputs : tensorflow.Tensor
            A tensor representing the tokenized input sequence.
        
        mask : tensorflow.Tensor, optional
            An existing mask, if any, to combine with the computed mask.
        
        Returns
        -------
        tensorflow.Tensor
            A mask tensor where padding tokens (zeros) are marked as False.
        """
        return tensorflow.math.not_equal(inputs, 0)
    
    def get_config(self):
        """
        Returns the configuration of the PositionalEmbedding layer, which includes
        the sequence length, input dimension, and output dimension for reinitialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the layer, including:
            - 'sequence_length': The length of the input sequences.
            - 'input_dim': The size of the vocabulary.
            - 'output_dim': The embedding dimension.
        """
        config = super().get_config()
        config.update({
            'output_dim' : self.output_dim,
            'sequence_length' : self.sequence_length,
            'input_dim' : self.input_dim,
        })
        return config
import tensorflow
import base_model
import helper_layers

class MLP(base_model.BaseModel):
    '''Multi Layer Perceptron
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, num_train_items, batch_size, n_epochs, cls_or_reg, write_summary, 
                 initial_lr, callbacks_state, log_dir, sub_dir, dense_units:list[int]=[128,64, 1,], 
                 dropout_units:list[float]=[.3, .5], dense_activations:list[str]=['relu', 'sigmoid',], 
                 *args, **kwargs):
        '''initialize the densestack object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
            dim : int
            usually the same value as the number of max tokens specified for the text vectorization
            layer.
            
            dense_units : list[int] 
            Default value is [128, 64, 1,]. specifies the number of units for the intermediary dense
            layers and the number of units for the last dense layer respectively.
            
            dropout_units : list[float] 
            Default value is [.3, .5]. specifies the dropout values for the dropout layers.
            
            dense_activations : list[str]
            Default value is ['relu', 'sigmoid']. activation functions supplied to the dense layers;
            First one is input to the intermediary layer and the second one for the last dense layer.
            
            kwargs : iterable, optional
            anyother user specified keywords args passed to the model.
        '''
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg, write_summary_state=write_summary, 
                         initial_lr=initial_lr, log_dir=log_dir, sub_dir=sub_dir, 
                         callbacks_state=callbacks_state, classification_type=cls_t)
        self.dense_units=dense_units
        self.dropout_units=dropout_units
        self.dense_activations=dense_activations
        self.dense_1=tensorflow.keras.layers.Dense(dense_units[0], 
                                                   activation=dense_activations[0])
        self.dropout_1=tensorflow.keras.layers.Dropout(dropout_units[0])
        self.dense_2=tensorflow.keras.layers.Dense(dense_units[1], 
                                                   activation=dense_activations[0])
        self.dropout_2=tensorflow.keras.layers.Dropout(dropout_units[-1])
        self.dense_3=tensorflow.keras.layers.Dense(dense_units[-1], 
                                                   activation=dense_activations[-1])
        
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
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
            }
        )
        return config

class BasicDense(base_model.BaseModel):
    '''BasicDense is a simple model of a dense and a dropout layer
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, num_train_items, batch_size, n_epochs, cls_or_reg, write_summary, 
                 initial_lr, callbacks_state, log_dir, sub_dir, dense_units:list[int]=[16, 1,], 
                 dropout_units:float=.5, dense_activations:list[str]= ['relu', 'sigmoid',],
                 *args, **kwargs):
        '''initialize the densestack object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
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
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg, write_summary_state=write_summary, 
                         initial_lr=initial_lr, log_dir=log_dir, sub_dir=sub_dir, 
                         callbacks_state=callbacks_state, classification_type=cls_t)
        self.dense_units=dense_units
        self.dropout_units=dropout_units
        self.dense_activations=dense_activations
        self.dense_1=tensorflow.keras.layers.Dense(dense_units[0], 
                                                   activation=dense_activations[0])
        self.dropout_1=tensorflow.keras.layers.Dropout(dropout_units)
        self.dense_2=tensorflow.keras.layers.Dense(dense_units[-1], 
                                                   activation=dense_activations[1])
        
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
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
            }
        )
        return config

class BasicEmbedding(base_model.BaseModel):
    '''BasicEmbedding is a simple model of embedding, global average pooling, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, num_train_items, batch_size, n_epochs, cls_or_reg, 
                 write_summary, initial_lr, callbacks_state, log_dir, sub_dir, 
                 dim=20000, o_dim:int=128, dense_units:list[int]=[16, 1,], 
                 dropout_units:float=.5, dense_activations:list[str]= ['relu', 'sigmoid'], 
                 mask_zero:bool=False, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
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
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg,
                         write_summary_state=write_summary, initial_lr=initial_lr,
                         log_dir=log_dir, sub_dir=sub_dir, callbacks_state=callbacks_state,
                         classification_type=cls_t)
        self.dim=dim
        self.o_dim=o_dim
        self.dense_units=dense_units
        self.dropout_units=dropout_units
        self.dense_activations=dense_activations
        self.mask_zero=mask_zero
        self.embedding=tensorflow.keras.layers.Embedding(input_dim=dim, 
                                                         output_dim=o_dim, 
                                                         mask_zero=self.mask_zero)
        self.glob_avg_pool=tensorflow.keras.layers.GlobalAveragePooling1D()
        self.dense_1=tensorflow.keras.layers.Dense(dense_units[0], 
                                                   activation=dense_activations[0])
        self.dropout_1=tensorflow.keras.layers.Dropout(dropout_units)
        self.dense_2=tensorflow.keras.layers.Dense(dense_units[-1], 
                                                   activation=dense_activations[-1])
        
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
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
            }
        )
        return config
    
class BasicLSTM(base_model.BaseModel):
    '''BasicLSTM is a simple model of embedding, bidirectional, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, num_train_items, batch_size, n_epochs, cls_or_reg, write_summary, 
                 initial_lr, callbacks_state, log_dir, sub_dir, dim=20000, o_dim:int=256, 
                 dense_units:int=1, dropout_units:float=.5, dense_activations:str='sigmoid', 
                 mask_zero:bool=False, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
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
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg,
                         write_summary_state=write_summary, initial_lr=initial_lr,
                         log_dir=log_dir, sub_dir=sub_dir, callbacks_state=callbacks_state,
                         classification_type=cls_t)
        self.dim=dim
        self.o_dim=o_dim
        self.dense_units=dense_units
        self.dropout_units=dropout_units
        self.dense_activations=dense_activations
        self.mask_zero=mask_zero
        self.embedding=tensorflow.keras.layers.Embedding(input_dim=dim, 
                                                         output_dim=o_dim,
                                                         mask_zero=self.mask_zero)
        self.bidir_lstm=tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32))
        self.dense=tensorflow.keras.layers.Dense(dense_units, 
                                                 activation=dense_activations)
        self.dropout=tensorflow.keras.layers.Dropout(dropout_units)
        
        
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
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
            }
        )
        return config

class Transformer(base_model.BaseModel):
    '''Transformer is a simple model of embedding, transformer encoder, global max pooling,
    dense, and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name:str, num_train_items, batch_size, 
                 n_epochs, cls_or_reg, write_summary, 
                 initial_lr, callbacks_state, log_dir, sub_dir,
                 enc_name:str=None, dense_stack_name:str=None, 
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
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg,
                         write_summary_state=write_summary, initial_lr=initial_lr,
                         log_dir=log_dir, sub_dir=sub_dir, callbacks_state=callbacks_state,
                         classification_type=cls_t)
        self.name=name
        self.enc_name=enc_name
        self.dense_stack_name=dense_stack_name
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.dense_dim=dense_dim
        self.dropout_val=dropout_val
        self.dense_unit=dense_unit
        self.dense_activation=dense_activation
        self.embed_1=tensorflow.keras.layers.Embedding(vocab_size, 
                                                       embed_dim,
                                                       input_shape=(None,))
        self.trf_enc=helper_layers.TransformerEncoder(embed_dim, 
                                                      dense_dim, 
                                                      num_heads,
                                                      name=enc_name, 
                                                      dense_name=dense_stack_name)
        self.globmxp=tensorflow.keras.layers.GlobalMaxPooling1D()
        self.dropout=tensorflow.keras.layers.Dropout(self.dropout_val)
        self.dense=tensorflow.keras.layers.Dense(self.dense_unit, 
                                                 activation=self.dense_activation)
    
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
    

class LSTM(base_model.BaseModel):
    '''BasicLSTM is a simple model of embedding, bidirectional, dense, 
    and dropout layers.
    
    Methods
    -------
    call(self, inputs): takes batches of string data and target values to attempt a prediction.
    '''
    def __init__(self, name, num_train_items, batch_size, n_epochs, cls_or_reg, 
                 write_summary, initial_lr, callbacks_state, log_dir, sub_dir, dim=20000, 
                 o_dim:int=256, dense_units:list[int]=[32, 1], dropout_units:float=.5, 
                 dense_activations:str='sigmoid', seq_len:int=None, num_heads:int=2, *args, **kwargs):
        '''initialize the embedding model object.
        
        Parameters
        ----------
            name : str
            user specified name for the model obj.
            
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
        if cls_or_reg=='cls':
            if self.last_dense_unit==1:
                cls_t='binary'
            else:
                cls_t='multiclass'
        else:
            cls_t='reg'
        super().__init__(model_name=name, num_train_items=num_train_items, batch_size=batch_size,
                         n_epochs=n_epochs, cls_or_reg=cls_or_reg,
                         write_summary_state=write_summary, initial_lr=initial_lr,
                         log_dir=log_dir, sub_dir=sub_dir, callbacks_state=callbacks_state,
                         classification_type=cls_t)
        self.dim=dim
        self.o_dim=o_dim
        self.dense_units=dense_units
        self.dropout_units=dropout_units
        self.dense_activations=dense_activations
        self.seq_len=seq_len
        self.num_heads=num_heads
        self.positional_embed=helper_layers.PositionalEmbedding(self.seq_len, 
                                                                self.dim, 
                                                                self.o_dim)
        self.encoder=helper_layers.TransformerEncoder(self.o_dim, 
                                                      self.dense_units[0], 
                                                      num_heads)
        self.glob_max_pool=tensorflow.keras.layers.GlobalMaxPooling1D()
        self.dense=tensorflow.keras.layers.Dense(dense_units[1], 
                                                 activation=dense_activations)
        self.dropout=tensorflow.keras.layers.Dropout(dropout_units)
        
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
        i = self.positional_embed(inputs)
        i = self.encoder(i)
        i = self.glob_max_pool(i)
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
                'dim' : self.dim,
                'o_dim' : self.o_dim,
                'dense_units' : self.dense_units,
                'dropout_units' : self.dropout_units,
                'dense_activations' : self.dense_activations
            }
        )
        return config
    

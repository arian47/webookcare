import tensorflow
import shutil
import pathlib
import webookcare.custom_metrics
import tensorflow_docs

class CommonConfs():
    """a class to form common configurations needed across different models.
    
    Methods
    -------
    initialize_params(self): sets up the learning scheduler, optimizer, callbacks, metrics, epochs,
        and loss function depending on the type of classification needed.
    
    """
    def __init__(self, name:str, num_train_items:int, batch_size:int, 
                 classification_type:str, n_epochs:int=10, cls_or_reg:str='reg',
                 write_summary:bool=True, initial_lr:float=1e-4, 
                 log_dir:str='tensorboard_logs', sub_dir:str='', 
                 callbacks_stat:bool=True, *args, **kwargs):
        """initialize a commonconf object holding common configuration values
        across different models.
        
        Parameters
        ----------
            name : str
            user specified name.
            
            num_train_items : int
            number of elements within the training dataset needed
            to determine the learning schedulers decay steps.
            
            batch_size : int
            batch size for the datasets passed in for the training.
            
            classification_type : str
            'binary' or 'multiclass' classification used for the model
            training.
            
            n_epochs : int
            number of the training epochs.
            
            cls_or_reg : str
            'cls' or 'reg' classification or regression for the training.
            
            write_summary : bool
            Default is True. whether or not to create a summary writer.
            
            initial_lr : float
            Default value is 1e-4. the initial value used for the learning rate.
            
            log_dir : str
            Default value is 'tensorboard_logs'. the dir to write the summary info.
            
            sub_dir : str
            Default value is ''. sub dir for each training phase.
            
            callbacks_stat : bool
            Default is True. whether to use callbacks for the training process.
        """
        self.name = name
        self.num_train_items = num_train_items
        self.batch_size = batch_size
        self.classification_type = classification_type
        self.n_epochs = n_epochs
        self.cls_or_reg = cls_or_reg
        self.ws_fl = write_summary
        self.initial_lr = initial_lr
        self.log_dir = log_dir
        self.sub_dir = sub_dir
        self.callbacks_stat = callbacks_stat
    
    def initialize_params(self):
        """sets up the learning scheduler, optimizer, callbacks, metrics, epochs,
        and loss function depending on the type of classification needed.
        
        Returns
        -------
            None
        """
        self.lr_schedule = \
            tensorflow.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_lr, 
                decay_steps=(self.num_train_items//self.batch_size), 
                decay_rate=0.96, 
                staircase=True)
        
        if self.cls_or_reg == 'cls':
            if self.classification_type=='binary':
                monitor_val = 'val_binary_accuracy'
                self.loss_fn = tensorflow.keras.losses.BinaryCrossentropy()
                self.loss_metric = tensorflow.keras.metrics.BinaryCrossentropy(name="binary_loss")
                self.acc_metric = tensorflow.keras.metrics.BinaryAccuracy(name="binary_accuracy")
            elif self.classification_type=='multiclass':
                monitor_val = 'val_categorical_accuracy'
                self.loss_fn = tensorflow.keras.losses.CategoricalCrossentropy()
                self.loss_metric = tensorflow.keras.metrics.CategoricalCrossentropy(name="categorical_loss")
                self.acc_metric = tensorflow.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
        else:
            monitor_val = 'val_mse'
            self.loss_fn = tensorflow.keras.losses.MeanSquaredError()
            self.loss_metric = tensorflow.keras.metrics.RootMeanSquaredError()
            self.acc_metric = webookcare.custom_metrics.R2Score()
            
        
        self.optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        
        if self.callbacks_stat:
            self.callbacks = [
                tensorflow_docs.modeling.EpochDots(),
                tensorflow.keras.callbacks.EarlyStopping(
                    monitor=monitor_val,
                    patience=5, 
                    restore_best_weights=True
                ),
            ]
        
        if self.ws_fl:
            tmp_dir = pathlib.Path(f'./{self.log_dir}/{self.sub_dir}/{self.name}')

            shutil.rmtree(tmp_dir, 
                          ignore_errors=True)
            
            # Initialize TensorBoard writer
            self.summary_writer = tensorflow.summary.create_file_writer(str(tmp_dir))
        
            self.callbacks.append(
                tensorflow.keras.callbacks.TensorBoard(log_dir=tmp_dir, 
                                                       histogram_freq=1,
                                                       write_graph=True)
            )

class BaseModel(tensorflow.keras.models.Model):
    '''BaseModel Representing basic common operations.'''
    def __init__(self, model_name, num_train_items:int, batch_size:int=32,
                 n_epochs:int=10, cls_or_reg:str='reg', write_summary_state:bool=True,
                 initial_lr:float=1e-4, log_dir:str='tensorboard_logs', callbacks_state:bool=True,
                 sub_dir:str='', classification_type:str='reg', *args, **kwargs):
        '''
        Parameters
        ----------
            num_train_items : int
            number of elements within the training dataset needed
            to determine the learning schedulers decay steps.
            
            batch_size : int
            batch size for the datasets passed in for the training.
            
            name : str
            Default value is None. user specified name for the model obj.    
            
            n_epochs : int
            Default value is 10. number of the training epochs.
            
            cls_or_reg : str
            'cls' or 'reg' classification or regression for the training.
            
            write_summary_state : bool
            Default value is True. whether to write summaries during training and validation
            steps or not for use with tensorboard.
            
            initial_lr : float
            Default value is 1e-4. initial value for the learning rate passed to the optimizer.
            
            log_dir : str
            Default value is 'tensorboard_logs'. main dir for writing logs.
            
            callbacks_state : bool
            Default value is True. whether to use callbacks or not.
            
            sub_dir : str
            Default value is ''. sub dir for writing logs.
            
            classification_type : str
            Default value is 'reg'. classification or reg for the task of training.
        '''
        super().__init__(model_name)
        self.model_name=model_name
        self.num_train_items=num_train_items
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.cls_or_reg=cls_or_reg
        self.write_summary_state=write_summary_state
        self.initial_lr=initial_lr
        self.log_dir=log_dir
        self.sub_dir=sub_dir
        self.callbacks_state=callbacks_state
        self.classification_type = classification_type
        self.conf=CommonConfs(model_name, num_train_items=self.num_train_items,
                              batch_size=self.batch_size, classification_type=self.classification_type,
                              n_epochs=self.n_epochs, cls_or_reg=self.cls_or_reg,
                              write_summary=self.write_summary_state,
                              initial_lr=self.initial_lr, log_dir=self.log_dir,
                              sub_dir=self.sub_dir, callbacks_state=self.callbacks_state)
        self.conf.initialize_params()
        
        def close_summary_writer(self):
            self.conf.summary_writer.close() 

    def get_config(self):
        '''builds the dictionary which hold necessary parameters to reconstruct
        model obj.
        
        Returns
        -------
            dict obj
        '''
        config = super().get_config()
        config.update(
            {
                'model_name' : self.model_name,
                'num_train_items' : self.num_train_items,
                'batch_size' : self.batch_size,
                'n_epochs' : self.n_epochs,
                'cls_or_reg' : self.cls_or_reg,
                'write_summary' : self.write_summary_state,
                'initial_lr' : self.initial_lr,
                'log_dir' : self.log_dir,
                'sub_dir' : self.sub_dir,
                'callbacks_state' : self.callbacks_state,
                'classification_type' : self.classification_type
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        if not optimizer:
            self.optimizer = self.conf.optimizer
        elif optimizer:
            self.optimizer = optimizer
        if not loss and not metrics:
            self.loss_fn = self.conf.loss_fn
            self.loss_metric = self.conf.loss_metric
            self.acc_metric = self.conf.acc_metric

        elif loss and metrics:
            self.loss_fn = loss
            self.acc_metric = metrics
        
        super().compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[self.loss_metric, 
                     self.acc_metric]
        )
        
    def compute_loss(self, real_data, generated_data):
        loss = self.loss_fn(real_data, 
                            generated_data)
        return loss
    
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
        
    def compute_metrics(self, real_data, generated_data):
        self.loss_metric.update_state(real_data, 
                                      generated_data)
        self.acc_metric.update_state(real_data, 
                                     generated_data)
        return {m.name : m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_metric,
                self.acc_metric,]
    
    def train_step(self, inputs):
        input_, target = inputs
        with tensorflow.GradientTape() as tape:
            output = self(input_, 
                          training=True) # will return batch of (NUM_UNIQUE_TERM,) shaped vectors
            loss = self.compute_loss(target, 
                                     output)
        
        gradients = tape.gradient(loss, 
                                  self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, 
                                           self.trainable_weights))
        metrics = self.compute_metrics(target, 
                                       output)
        
        if self.write_summary_state:
            # Log metrics to TensorBoard
            with self.conf.summary_writer.as_default():
                tensorflow.summary.scalar(f'{self.loss_metric.name}', 
                                          loss, 
                                          step=self.optimizer.iterations)
                tensorflow.summary.scalar(f'{self.acc_metric.name}', 
                                          metrics[self.acc_metric.name], 
                                          step=self.optimizer.iterations)
        
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
        output = self(input_, 
                      training=False)
        loss = self.compute_loss(target, 
                                 output)
        metrics = self.compute_metrics(target, 
                                       output)
        
        if self.write_summary_state:
            with self.conf.summary_writer.as_default():
                tensorflow.summary.scalar(f'val_{self.loss_metric.name}', 
                                          loss, 
                                          step=self.optimizer.iterations)
                tensorflow.summary.scalar(f'val_{self.acc_metric.name}', 
                                          metrics[self.acc_metric.name], 
                                          step=self.optimizer.iterations)
        return metrics
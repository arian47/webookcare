import tensorflow

class ProcessData:
    def __init__(self, input_data, tvec_mt:int=20000, tvec_om:str='int', 
                 ngrams:int=1, *args, **kwargs):
        self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
                                                              output_mode=tvec_om)
        self.tvec.adapt(input_data)

class VectorizePrediction:
    def __init__(self, input_data, tvec_mt:int=20000, tvec_om:str='int', 
                 ngrams:int=1, vocab:list[str]=None, pad_to_max_tokens:bool=False,
                 output_sequence_length:int=None, *args, **kwargs):
        self.tvec = tensorflow.keras.layers.TextVectorization(max_tokens=tvec_mt,
                                                              output_mode=tvec_om,
                                                              vocabulary=vocab,
                                                              pad_to_max_tokens=pad_to_max_tokens,
                                                              output_sequence_length=output_sequence_length)
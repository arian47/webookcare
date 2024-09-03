import numpy
import tensorflow
import pathlib
import re
import collections.abc

class PostProcess:
    
    def remove_nones(self, data:list[str]=None, labels:list[str]=None):
        invalid_terms = [None, 'nan', numpy.nan, 'NaN',]
        while True:
            indices = []
            # if data is not None and labels is not None:
            assert type(data) == list and type(labels) == list, \
            'data or labels is not a list!'
            if type(data) == list and type(labels) == list:
                # assert len(data) == len(labels), 'unmatching length for data and labels'
                if len(data) != len(labels):
                    print('during post process unmatching lengths found for data and labels!')
                    min_len = min(len(data), len(labels))
                    print(f'lengths do not match will try trimming to val {min_len}')
                    data = data[:min_len]
                    labels = labels[:min_len]
                for i in range(len(data)):
                    # data has to be a string
                    if data[i] in invalid_terms or type(data[i]) != str:
                        indices.append(i)
                    
                    # labels could be a nested list or string
                    if type(labels[i]) == list:
                        for x in labels[i]:
                            # need changes below only a place holder
                            if x in invalid_terms or type(x) != str:
                                indices.append(i)
                    elif not type(labels[i]) == list:
                        if labels[i] in invalid_terms or type(labels[i]) != str:
                            indices.append(i)
                indices = set(indices)
                data = [data[i] for i in range(len(data)) if i not in indices]
                labels = [labels[i] for i in range(len(labels)) if i not in indices]
                return data, labels
            
            # elif type(data) == list and type(labels) != list:
            #     for i in range(len(data)):
            #         if data[i] in invalid_terms or type(data[i]) != str:
            #             if i not in indices:
            #                 indices.append(i)
            #     data = [data[i] for i in range(len(data)) if i not in indices]
            #     return data
            
            # elif type(labels) == list and type(data) != list:
            #     for i in range(len(labels)):
            #         if labels[i] in invalid_terms or type(labels[i]) != str:
            #             if i not in indices:
            #                 indices.append(i)
            #     labels = [labels[i] for i in range(len(labels)) if i not in indices]
            #     return labels
    
    # TODO: might need to change some stuff for list handling later
    def create_ngrams(self, data, ngrams_val=2):
        '''defaults to creating bigrams. words are separated with whitespaces only.'''
        ngrams_vocab = []
        ngrams = []
        if type(data) == list:
            for i in data:
                for x in i:
                    tokens = x.split()
                    ngrams_tmp = [' '.join(tokens[j:j+ngrams_val]) for j in range(len(tokens)-ngrams_val+1)]
                    ngrams_vocab.extend(ngrams_tmp)
                    ngrams.append(ngrams_tmp)
        else:    
            for i in data:
                tokens = i.split()
                ngrams_tmp = [' '.join(tokens[j:j+ngrams_val]) for j in range(len(tokens)-ngrams_val+1)]
                ngrams_vocab.extend(ngrams_tmp)
                ngrams.append(ngrams_tmp)
        return ngrams_vocab, ngrams
    
    # TODO: might need to change some stuff for list handling late     
    def pad_ngrams(self, ngrams, vocab):
        padding_token = '<PAD>'
        vocab = sorted(set(vocab + [padding_token]))
        ngrams_padded = tensorflow.keras.preprocessing.sequence.pad_sequences(
            ngrams, dtype=object, padding='post', value=padding_token)
        return ngrams_padded, vocab
    
    # TODO: might need to change some stuff for list handling late
    def multihot_encode(self, ngrams, vocab):
        lookup = tensorflow.keras.layers.StringLookup(vocabulary=vocab, output_mode='multi_hot')
        multi_hot_encoded_data = lookup(ngrams)
        return multi_hot_encoded_data, lookup.get_vocabulary()

    def custom_pattern(data:list, patterns_file:pathlib.Path='terms_to_remove.txt', 
                       names_file:pathlib.Path='yob2023.txt'):
        # data = postprocess(data, patterns, names)
        # labels = postprocess(labels, patterns, names)
        assert data, 'Data is empty!'
        assert patterns_file.exists(), 'file not found!'
        
        with patterns_file.open('rt', encoding='utf-8') as fo:
            patterns = [i.strip() for i in fo]
            
        with names_file.open('rt', encoding='utf-8') as fo:
            names = [i.split(',', maxsplit=1)[0] for i in fo]
        
        tmp_data = []
        for i in data:
            if isinstance(i, str):
                tmp_str = ' '.join(i.split())
                tmp_data.append(tmp_str)
            elif i is None:
                tmp_data.append(None)
            elif isinstance(i, tuple):
                tmp_str = ' '.join(i).strip()
                tmp_data.append(tmp_str)
        
        pattern = re.compile('|'.join(patterns), flags=re.IGNORECASE)
        name_pattern = re.compile(r'\b(?:{})\b'.format('|'.join(map(re.escape, names))))
        
        flag = True
        while flag:
            tmp_data2 = []
            for i in tmp_data:
                if i is None:
                    tmp_data2.append(None)
                else:
                    tmp_data2.append(pattern.sub('', i))
                    
            flag = tmp_data2 != tmp_data
            tmp_data = tmp_data2
        
        
        tmp_data3 = []
        for i in tmp_data:
            if i is None:
                tmp_data3.append(None)
            else:
                tmp_data3.append(name_pattern.sub('', i))
        
        return tmp_data3
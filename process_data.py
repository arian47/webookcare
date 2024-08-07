import pathlib
import re
import shutil
import numpy
import tensorflow

class Preprocess:
    def __init__(self):
        self.files_pattern = r'duties.*list'
        
    def move_files(self, pattern:str, dir_:pathlib.Path, suffix, ndir:str):
        paths = list(dir_.glob(f'**/*{suffix}'))
        if paths:
            poi = []
            for i in paths:
                matches = re.findall(pattern, str(i), re.IGNORECASE)
                if matches:
                    poi.append(i)
        new_path = pathlib.Path('/'.join(dir_.parts[:-1]).replace('\\', '')).joinpath(f'{dir_.parts[-1]}_{ndir}')
        for i in poi:
            if not new_path.exists():
                new_path.mkdir()
            shutil.move(i, new_path)
        
    def remove_special_chars(self, data):
        pattern = r'[`~!@#$%^&*()\-_=+,.\/?;:\\|\[\]{}]'
        cleaned_data = [re.sub(pattern, ' ', i) for i in data if i not in (None, numpy.nan,)]
        return cleaned_data
    
    def custom_pattern(self, pattern, replacement, text):
        tmp_txt = re.sub(pattern, replacement, text)
        return tmp_txt
    
    def str_check(self, data):
        for i in data:
            if type(i) != str:
                print(f'{i} has non-str data')
                
    def str_only(self, data):
        tmp_data = []
        for i in data:
            if type(i) == str:
                tmp_data.append(i)
        return tmp_data
                
    def find_non_utf8(self, data):
        for i, text in enumerate(data):
            try:
                if type(text) == str:
                    text = str.encode(text)
                else: 
                    continue
            except UnicodeEncodeError as e:
                print(f"Non-UTF-8 character found in text at index {i}: {text[e.start:e.end]}")
                
    def clean_non_utf8(self, data):
        cleaned_data = [i.encode('utf-8', 'ignore').decode('utf-8', 'ignore') for i in data]
        cleaned_data = [re.sub(r'[^\x00-\x7F]+', '', i) for i in cleaned_data]
        return cleaned_data
    
    def retrieve_basic_info(self, data, info_needed):
        if info_needed == 'email':
            pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        elif info_needed == 'phone':
            pattern = r'\+?\d{1,2}?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}'
        elif info_needed == 'name':
            pattern = r'\b[A-Z][a-z]*\s[A-Z][a-z]*(?:\s[A-Z][a-z]*)?\b'
        else:
            print('requested pattern is not available!')
            return []
        return re.findall(pattern, data)
    
    def extract_info(self, data):
        return {
            'names' : self.retrieve_basic_info(data, 'name'),
            'emails' : self.retrieve_basic_info(data, 'email'),
            'phones' : self.retrieve_basic_info(data, 'phone')
        }
        
        
class PostProcess:
    
    def remove_nones(self, data:list[str]=None, labels:list[str]=None):
        # track = True
        invalid_terms = [None, 'nan', numpy.nan, 'NaN',]
        while True:
            indices = []
            # if data is not None and labels is not None:
            if type(data) == list and type(labels) == list:
                # assert len(data) == len(labels), 'unmatching length for data and labels'
                if len(data) != len(labels):
                    min_len = min(len(data), len(labels))
                    print(f'lengths do not match will try trimming to val {min_len}')
                    data = data[:min_len]
                    labels = labels[:min_len]
                for i in range(len(data)):
                    # if data[i] is None:
                    # if data[i] == None:
                    if data[i] in invalid_terms or type(data[i]) != str:
                        # if i not in indices:
                        indices.append(i)
                    # if labels[i] is None:
                    # if labels[i] == None:
                    if labels[i] in invalid_terms or type(labels[i]) != str:
                        # if i not in indices:
                        indices.append(i)
                # track = False
                indices = set(indices)
                data = [data[i] for i in range(len(data)) if i not in indices]
                labels = [labels[i] for i in range(len(labels)) if i not in indices]
                # indices = []
                # break
                return data, labels
            
            # elif data is not None and labels is None:
            elif type(data) == list and type(labels) != list:
                for i in range(len(data)):
                    # if data[i] is None:
                    # if data[i] == None:
                    if data[i] in invalid_terms or type(data[i]) != str:
                        if i not in indices:
                            indices.append(i)
                data = [data[i] for i in range(len(data)) if i not in indices]
                # track = False
                # indices = []
                # break
                return data
            
            # elif labels is not None and data is None:
            elif type(labels) == list and type(data) != list:
                for i in range(len(labels)):
                    # if labels[i] is None:
                    # if labels[i] == None:
                    if labels[i] in invalid_terms or type(labels[i]) != str:
                        if i not in indices:
                            indices.append(i)
                labels = [labels[i] for i in range(len(labels)) if i not in indices]
                # track = False
                # indices = []
                # break
                return labels
            # data = [data[i] for i in range(len(data)) if i not in indices]
            # labels = [labels[i] for i in range(len(labels)) if i not in indices]
        # return data, labels
    
    def create_ngrams(self, data, ngrams_val=2):
        '''defaults to creating bigrams. words are separated with whitespaces only.'''
        
        ngrams_vocab = []
        ngrams = []
        
        for i in data:
            tokens = i.split()
            ngrams_tmp = [' '.join(tokens[j:j+ngrams_val]) for j in range(len(tokens)-ngrams_val+1)]
            ngrams_vocab.extend(ngrams_tmp)
            ngrams.append(ngrams_tmp)
        return ngrams_vocab, ngrams
            
    def pad_ngrams(self, ngrams, vocab):
        padding_token = '<PAD>'
        vocab = sorted(set(vocab + [padding_token]))
        ngrams_padded = tensorflow.keras.preprocessing.sequence.pad_sequences(
            ngrams, dtype=object, padding='post', value=padding_token)
        return ngrams_padded, vocab
    
    def multihot_encode(self, ngrams, vocab):
        lookup = tensorflow.keras.layers.StringLookup(vocabulary=vocab, output_mode='multi_hot')
        multi_hot_encoded_data = lookup(ngrams)
        return multi_hot_encoded_data, lookup.get_vocabulary()
    
    
class ExtractData:
    def __init__(self) -> None:
        self.data_patterns = {
            'client care plan 1' :
                r'Duties\s*to\s*Perform\s*Notes(.*?)Terms\s*of\s*Plan',
            'occupational therapy services 2' : 
                r'Notes:(.*?)Please\s*make\s*invoices\s*for\s*above\s*items\s*payable\s*to:',
            'comunity therapists 3' : [[
                r'Vanessa,(.*?)Laila',
                r'Medical\s*orders/Instructions:(.*?)PERSONAL\s*ATTENDANT\s*CARE',
                ]],
            'care plan 4' :
                r'Condition(.*?)Characteristics',
            'client care plan 5' :
                r'CONDITION(.*?)HEALTHCARE\s*PROFESSIONALS',
            'care plan 6' :
                r'General\s*Notes(.*?)Extra\s*Information',
            'home healthcare worker care plan 7' : 
                r'Comments:(.*?)Safety/Risk:',
            'care plan 8': 
                r'Condition(.*?)Characteristic'
            
        } # has 9 rules in total worse-case scenario
        
            # 'healthcare_worker_plan' : r'.*healthcare\s*worker\s*plan.*',
            # 'homemaking_duties' : r'.*homemaking\s*duties.*',
            # 'client_handbook' : r'.*client\s*handbook.*',
            # 'duties_list' : r'.*duties\s*list.*',
        self.labels_patterns = {
            'comunity therapists 3' : [
                [r'HOME\s*MAKING\s*SERVICES(.*?)ADDITIONAL\s*ACTIVITIES',
                 r'HOME\s*MAKING\s*SERVICES(.*?)Vanessa',
                 r'HOME\s*MAKING\s*SERVICES(.*?)Laila',
                 r'HOME\s*MAKING\s*SERVICES.*'],
                [r'PERSONAL\s*ATTENDANT\s*CARE(.*?)ADDITIONAL\s*ACTIVITIES',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*?)Hi\s*Vanessa',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*?)Terms\s*of\s*Plan',
                r'PERSONAL\s*ATTENDANT\s*CARE(.*)',] # to be thought about
            ],
            'care plan 4' :
                r'Client\s*Care\s*Plan(.*?)Authorization\s*Care\s*Plans'
            ,
            'client care plan 5': [[
                r'DUTIES\s*TO\s*PERFORM(.*?)Duties\s*to\s*Perform\s*Notes'
                r'DUTIES\s*TO\s*PERFORM(.*?)AUTHORIZATIONS',
            ]],
            'home healthcare worker care plan 7': [[ 
                r'✓(.*?)(prn|✓|2X/wk|1X/wk|Daily)',
                r'✓(.*?)ADL’s'
            ]],
            'care plan 8' : 
                r'.*Duties\s*to\s*Perform(.*?)Current\s*Authorizations.*',
            'occupational therapy services 2' : 
                r'.*Duties:(.*?)(Client’s information).*'
        }
    
    # TODO: move home_healthcare_worker_care_plan, homemaking_duties,
    # client_handbook, duties_list files to a separate folder
    def auto_act(self):
        pass
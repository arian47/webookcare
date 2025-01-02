import re
import pathlib
import shutil
import numpy
import collections.abc

class Preprocess:
    """Simple preprocessing for the textual data
    """
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
        if isinstance(data, collections.abc.Iterable):
            # cleaned_data = [re.sub(pattern, ' ', j) for i in data for j in i if i not in (None, numpy.nan,)]
            cleaned_data = [[re.sub(pattern, '', j) for j in i if re.sub(pattern, '', j)]
                            for i in data]
            # cleaned_data = []
            # for i in data:
            #     tmp_data = []
            #     for j in i:
            #         ss = re.sub(pattern, '', j)
            #         if ss:
            #             tmp_data.append(ss)
            #     cleaned_data.append(tmp_data)
        elif isinstance(data, str):
            cleaned_data = [re.sub(pattern, ' ', i) for i in data if i not in (None, numpy.nan,)]
        else:
            raise Exception('Unrecognized type!')
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
        if isinstance(data, list):
            if all(isinstance(i, list) for i in data):
                cleaned_data = [[j.encode('utf-8', 'ignore').decode('utf-8', 'ignore') for j in i] 
                                for i in data]
                cleaned_data = [[re.sub(r'[^\x00-\x7F]+', '', j) for j in i] 
                                for i in cleaned_data]
            else:
                cleaned_data = [
                    re.sub(r'[^\x00-\x7F]+', '', i.encode('utf-8', 'ignore').decode('utf-8', 'ignore')) 
                    for i in data]
        elif isinstance(data, str):
            cleaned_data = [i.encode('utf-8', 'ignore').decode('utf-8', 'ignore') for i in data]
            cleaned_data = [re.sub(r'[^\x00-\x7F]+', '', i) for i in cleaned_data]
        else:
            raise Exception('type not reognized!')
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
        
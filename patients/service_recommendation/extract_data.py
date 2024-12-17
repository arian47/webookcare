import pandas

#### old data pdfs

### First Run
# parent_dir = pathlib.Path(
#     "data/old data"
# )
# dirs = [i for i in parent_dir.iterdir() if i.is_dir()]
# foi = [j for i in dirs for j in i.rglob('**/*.pdf')]

# def extract_text_from_file(reader, patterns):
#     tmp_data = []
#     tmp_text = ' '
#     for j in range(len(reader.pages)):
#         tmp_text += reader.pages[j].extract_text()
#     tmp_text = tmp_text.strip().replace('\n', ' ')
#     flags = re.IGNORECASE | re.DOTALL

#     def process_patterns(patterns, text):
#         for pattern in patterns:
#             matches = re.findall(pattern, text, flags)
#             if matches:
#                 return matches
#         return None

#     matched = False
#     for key, pattern_set in patterns.items():
#         if isinstance(pattern_set, list):
#             for sub_patterns in pattern_set:
#                 if isinstance(sub_patterns, list):
#                     matches = process_patterns(sub_patterns, tmp_text)
#                 else:
#                     matches = re.findall(sub_patterns, tmp_text, flags)
#                 if matches:
#                     tmp_data.extend(matches)
#                     matched = True
#                     break
#         elif isinstance(pattern_set, str):
#             if key == 'client care plan 1':
#                 ignore_terms = ['Bathing \\(bath\\)', 'Dressing \\(lower\\)', 
#                                 'Wheelchair', 'Hair Care', 'Skin Care', 'Falls', 'Morning Meds']
#                 ignore_pattern = r'\b(?:{})\b'.format('|'.join(ignore_terms))
#                 clean_service_text = re.sub(ignore_pattern, '', tmp_text, flags)
#                 matches = re.findall(pattern_set, clean_service_text, flags)
#             else:
#                 matches = re.findall(pattern_set, tmp_text, flags)
#             if matches:
#                 tmp_data.extend(matches)
#                 matched = True
#                 break
#         else:
#             raise Exception('Not possible!')

#     if not matched:
#         tmp_data.append(None)

#     return tmp_data

# data = []
# labels = []
# cpp = ExtractData()
# for file in foi:
#     reader = PyPDF2.PdfReader(file)
#     if reader.is_encrypted:
#         try:
#             reader.decrypt('client')
#             data.extend(extract_text_from_file(reader, cpp.data_patterns))
#             labels.extend(extract_text_from_file(reader, cpp.labels_patterns))
#         except PyPDF2.errors.FileNotDecryptedError:
#             print(f'password for {file.stem} was incorrect')
#     else:
#         data.extend(extract_text_from_file(reader, cpp.data_patterns))
#         labels.extend(extract_text_from_file(reader, cpp.labels_patterns))

# def postprocess(data:list, patterns_file:pathlib.Path, names_file:pathlib.Path):
#     assert data, 'Data is empty!'
#     assert patterns_file.exists(), 'file not found!'
    
#     with patterns_file.open('rt', encoding='utf-8') as fo:
#         patterns = [i.strip() for i in fo]
        
#     with names_file.open('rt', encoding='utf-8') as fo:
#         names = [i.split(',', maxsplit=1)[0] for i in fo]
    
#     tmp_data = []
#     for i in data:
#         if isinstance(i, str):
#             tmp_str = ' '.join(i.split())
#             tmp_data.append(tmp_str)
#         elif i is None:
#             tmp_data.append(None)
#         elif isinstance(i, tuple):
#             tmp_str = ' '.join(i).strip()
#             tmp_data.append(tmp_str)
    
#     pattern = re.compile('|'.join(patterns), flags=re.IGNORECASE)
#     name_pattern = re.compile(r'\b(?:{})\b'.format('|'.join(map(re.escape, names))))
    
#     flag = True
#     while flag:
#         tmp_data2 = []
#         for i in tmp_data:
#             if i is None:
#                 tmp_data2.append(None)
#             else:
#                 tmp_data2.append(pattern.sub('', i))
                
#         flag = tmp_data2 != tmp_data
#         tmp_data = tmp_data2
    
    
#     tmp_data3 = []
#     for i in tmp_data:
#         if i is None:
#             tmp_data3.append(None)
#         else:
#             tmp_data3.append(name_pattern.sub('', i))
    
#     return tmp_data3

# patterns = pathlib.Path('terms_to_remove.txt')
# names = pathlib.Path("data/names/yob2023.txt")

# data = postprocess(data, patterns, names)
# labels = postprocess(labels, patterns, names)

# conn = sqlite3.connect('data_labels.db')
# cursor = conn.cursor()
# cursor.execute('CREATE TABLE IF NOT EXISTS DataLabels (id INTEGER PRIMARY KEY, data TEXT, label TEXT)')

# for i, j in zip(data, labels):
#     cursor.execute('INSERT INTO DataLabels (data, label) VALUES (?, ?)', (i, j))
    
# conn.commit()

# df = pandas.DataFrame({'Data': data})
# df.to_csv('data.csv', index=False)
# df = pandas.DataFrame({'Labels': labels})
# df.to_csv('labels.csv', index=False)

# numpy.save('data.npy', numpy.array(data))
# numpy.save('labels.npy', numpy.array(labels))

### Next Runs

# # Access data and labels
# cursor.execute('SELECT data, label FROM DataLabels')
# rows = cursor.fetchall()
# for row in rows:
#     print(row[0])  # Output: data1, data2, data3 (one per iteration)
#     print(row[1])  # Output: label1, label2, label3 (one per iteration)

# conn.close()

train_data_od_path = "data.csv"
train_labels_od_path = "labels.csv"

train_data_od = pandas.read_csv(train_data_od_path).to_numpy(na_value=None).flatten().tolist()
train_labels_od = pandas.read_csv(train_labels_od_path).to_numpy(na_value=None).flatten().tolist()

# train_labels_od = preo.remove_special_chars(train_labels_od)
# train_data_od, train_labels_od = pp.remove_nones(
#       train_data_od, train_labels_od
# )
# train_data_od = preo.clean_non_utf8(train_data_od)
# train_labels_od = preo.clean_non_utf8(train_labels_od)
# (train_labels_od_ng_vocab,
# train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 2)
# (train_labels_od_ngrams_padded,
#  train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
#      train_labels_od_ngrams,
#      train_labels_od_ng_vocab
#  )
# train_labels_od_ngrams_padded_mh = pp.multihot_encode(
#     train_labels_od_ngrams_padded,
#     train_labels_od_ng_vocab_padded
# )
# assert len(train_data_od) == len(train_labels_od)


# NUM_UNIQUE_TERMS = len(set(train_labels_od_ng_vocab_padded))
# print(NUM_UNIQUE_TERMS,
      # train_labels_od_ngrams_padded_mh.shape)
# NUM_UNIQUE_TERMS = train_labels_od_ngrams_padded_mh.shape[-1]















### v1 data
#### first run
# file_path = 'D:/projects/github repos/webookcare/data/v1'\
#     '/care_seeker_data.numbers'
# doc = Document(file_path)

# sheets = doc.sheets
# tables = sheets[0].tables
# # rows = tables[0].rows()
# # table = tables[0]

# data = tables[0].rows(values_only=True)
# df = pandas.DataFrame(data[1:], columns=data[0])

# excl_file_path = 'D:/projects/github repos/webookcare/data/v1'\
#     '/care_seeker_data.xlsx'
# df.to_excel(excl_file_path)

# fpath = 'C:/users/aryan/onedrive/desktop/gg.csv'
# data_csv = df.to_csv(fpath)


#### next runs
excl_file_path = 'D:/projects/github repos/webookcare/data/v1'\
    '/care_seeker_data.xlsx'
df = pandas.read_excel(excl_file_path)

test_data = df[['care_recipient_details', 'care_recipient_interests_hobbies']]
test_data = test_data['care_recipient_details'] + test_data['care_recipient_interests_hobbies']
test_data = test_data.tolist()
test_labels = df['Services'].tolist()
# test_labels = preo.remove_special_chars(test_labels)
# test_data, test_labels = pp.remove_nones(test_data, test_labels)
# test_data = preo.clean_non_utf8(test_data)
# test_labels = preo.clean_non_utf8(test_labels)
# (test_labels_ng_vocab,
# test_labels_ngrams) = pp.create_ngrams(test_labels, 2)
# (test_labels_ngrams_padded,
#  test_labels_ng_vocab_padded) = pp.pad_ngrams(
#      test_labels_ngrams,
#      test_labels_ng_vocab
#  )
# (test_ngrams_padded_mh,
#  test_mh_vocab) = pp.multihot_encode(
#     test_labels_ngrams_padded,
#     test_labels_ng_vocab_padded
# )
# assert len(test_labels) == len(test_data)
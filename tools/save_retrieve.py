import sqlite3
import pandas
import numpy

def save_results(data, labels):
    conn = sqlite3.connect('data_labels.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS DataLabels (id INTEGER PRIMARY KEY, data TEXT, label TEXT)')

    for i, j in zip(data, labels):
        cursor.execute('INSERT INTO DataLabels (data, label) VALUES (?, ?)', (i, j))
        
    conn.commit()
    
    df = pandas.DataFrame({'Data': data})
    df.to_csv('data.csv', index=False)
    df = pandas.DataFrame({'Labels': labels})
    df.to_csv('labels.csv', index=False)

    numpy.save('data.npy', numpy.array(data))
    numpy.save('labels.npy', numpy.array(labels))
    
def retrieve_results():
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
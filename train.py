import pandas
import process_data
import numpy
import tensorflow
import helper_layers
import models
import pathlib
import nlpaug.augmenter.word as naw


preo = process_data.Preprocess()
pp = process_data.PostProcess()

paraphrased_sentences_path = pathlib.Path("paraphrased_sentences.npy").absolute().as_posix()
augmented_labels_path = pathlib.Path("augmented_labels.npy").absolute().as_posix()

model_save_path = pathlib.Path('MLP_int_1g_bigram_multihot_labels').absolute().as_posix()

def train_model():
    try:
        paraphrased_sentences = numpy.load(
            paraphrased_sentences_path).flatten().tolist()
        augmented_labels = numpy.load(
            augmented_labels_path).flatten().tolist()
    except OSError:
        train_data_od_path = pathlib.Path("data.csv").absolute().as_posix()
        train_labels_od_path = pathlib.Path("labels.csv").absolute().as_posix()
        excl_file_path = pathlib.Path('care_seeker_data.xlsx').absolute().as_posix()

        train_data_od = pandas.read_csv(
            train_data_od_path).to_numpy(na_value=None).flatten().tolist()
        train_labels_od = pandas.read_csv(
            train_labels_od_path).to_numpy(na_value=None).flatten().tolist()

        df = pandas.read_excel(excl_file_path)
        test_data = df[
            ['care_recipient_details', 
            'care_recipient_interests_hobbies']
            ]
        test_data = test_data['care_recipient_details'] + \
            test_data['care_recipient_interests_hobbies']
        test_data = test_data.tolist()
        test_labels = df['Services'].tolist()

        train_data_od.extend(test_data)
        train_labels_od.extend(test_labels)

        train_labels_od = preo.remove_special_chars(train_labels_od)
        train_data_od, train_labels_od = pp.remove_nones(
            train_data_od, train_labels_od
        )
        train_data_od = preo.clean_non_utf8(train_data_od)
        train_labels_od = preo.clean_non_utf8(train_labels_od)
        (train_labels_od_ng_vocab,
        train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 2)
        (train_labels_od_ngrams_padded,
        train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
            train_labels_od_ngrams,
            train_labels_od_ng_vocab
        )
        (train_labels_od_ngrams_padded_mh,
        train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
            train_labels_od_ngrams_padded,
            train_labels_od_ng_vocab_padded
        )
        assert len(train_data_od) == len(train_labels_od)

        NUM_UNIQUE_TERMS = train_labels_od_ngrams_padded_mh.shape[-1]
        
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="substitute")
        
        augmented_sentences = [aug.augment(i) for i in train_data_od for j in range(50)]
        df = pandas.DataFrame({'Augmented Sentences' : augmented_sentences})
        df.to_csv('augmented_sentences.csv', 
                index=False)
        numpy.save('paraphrased_sentences.npy', 
                numpy.array(augmented_sentences))
        
        augmented_labels = [i for i in train_labels_od for j in range(50)]
        df = pandas.DataFrame({'Augmented Labels' : augmented_labels})
        df.to_csv('augmented_labels.csv', 
                index=False)
        
        numpy.save('augmented_labels.npy', 
                numpy.array(augmented_labels))
        

    else:
        train_augmented_labels_od = preo.remove_special_chars(augmented_labels)
        train_augmented_data_od, train_augmented_labels_od = pp.remove_nones(
            paraphrased_sentences, train_augmented_labels_od
        )
        train_augmented_data_od = preo.clean_non_utf8(train_augmented_data_od)
        train_augmented_labels_od = preo.clean_non_utf8(train_augmented_labels_od)
        (train_augmented_labels_od_ng_vocab,
        train_augmented_labels_od_ngrams) = pp.create_ngrams(train_augmented_labels_od, 2)
        (train_augmented_labels_od_ngrams_padded,
        train_augmented_labels_od_ng_vocab_padded) = pp.pad_ngrams(
            train_augmented_labels_od_ngrams,
            train_augmented_labels_od_ng_vocab
        )
        (train_augmented_labels_od_ngrams_padded_mh,
        train_augmented_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
            train_augmented_labels_od_ngrams_padded,
            train_augmented_labels_od_ng_vocab_padded
        )
        assert len(train_augmented_data_od) == len(train_augmented_labels_od)

        o = helper_layers.ProcessData(train_augmented_data_od, 
                                    20000, 
                                    'int')
        train_data_od_t = tensorflow.constant(train_augmented_data_od) # tensor of strings
        train_data_od_t = o.tvec(train_data_od_t)

        numpy.save("train_data_int_1g_vocab.npy",
                o.tvec.get_vocabulary())
        
        model = models.MLP('MLP', 
                        dense_units=[8192, 
                                        4096, 
                                        train_augmented_labels_od_ngrams_padded_mh.shape[-1]
                                        ]
                        )
        model.compile()
        history = model.fit(x=train_data_od_t,
                            y=train_augmented_labels_od_ngrams_padded_mh,
                            validation_split=0.3,
                            epochs=10)
        
        tensorflow.saved_model.save(model, 
                                    model_save_path)

    # Load the saved model (optional if using SavedModel format)
    loaded_model = tensorflow.saved_model.load(model_save_path)

    # infer = loaded_model.signatures["serving_default"]
    return loaded_model
    
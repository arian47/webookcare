# webookcare

## Usage
In order to get the result of care tasks that are predicted the following procedure must be followed:
1- have the python up and running in your local environment
2- install necessary packages with requirement.txt file ('python -m pip install -r requirements.txt')
3- download "train_data_int_1g_vocab.npy" and "mh_labels_vocab.npy" in the same main dir as query.py
4- use the CLI from your shell executing "python query.py predict 'string 1' 'string 2' ..."

## Description
The prediction consists of weighted result of 
- text classification model used to predict bigram labels trained on data gathered from earlier version of application user data and other data gathered from pdf documents utilizing re matching patterns which is a multilabel classification task and bag of bigrams are used for labels info gathered the model chosen for this is an MLP model as not enough data was available for effective use of sequence models and text augmentation techniques were used to increase the amount of train data available









## More on text classification model
- methods thought to assist with paraphrasing:
  - Manually rewrite sentences to convey the same meaning using different words and structures
  - Replace words with their synonyms using a thesaurus or a library like NLTK.
  - Use predefined grammatical rules to restructure sentences.
  - Use paraphrasing tools like QuillBot, Paraphraser.io, or APIs provided by services like TextRazor
  - Back-Translation; translate the sentence to another language and then back to the original language like Google Translate API or MarianMT
  - models like BERT, GPT-3, T5, or Pegasus for paraphrasing Hugging Face Transformers
  - Libraries like nlpaug or textattack for text augmentation techniques
  - platforms like Amazon Mechanical Turk to have multiple people paraphrase the sentences


overfitting is clearly visible even with the basic MLP structure (consisting of dense(8192), dropout(.3), dense(4096), dropout(.5), dense(len(multi label vocabulary))

![int 1 gram MLP train data, bigram multihot labels](https://github.com/user-attachments/assets/63bf8ca2-8c37-4249-91b8-e2881d66622c)

![MLP int 2 gram train data, bigram multihot labels](https://github.com/user-attachments/assets/f2f72994-20c1-495a-b477-6d49557c68af)


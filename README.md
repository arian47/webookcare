# webookcare

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

# Webookcare

## Usage

The requests are sent to a host you choose to run the main FastAPI app on. Currently, only the **patient ensemble model** has been created. This model relies on a few sub-models, including:

- Care services recommender
- Credentials recommender
- Distance predictor
- Collaborative filtering (used for ranking based on reviews)

### To set up the server, follow these steps:

1. **Access the saved models**: Ensure you have access to each saved model for the underlying components (or the training data to train them).
2. **Save database credentials**: Store the necessary credentials to access your database in your local environment by creating a `.env` file. A script is provided to help with this.
3. **Set up Python on your host machine**: Make sure Python is installed and set up correctly.
4. **Install required packages**: Install the required dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```
5. **Launch the FastAPI app**: Run the FastAPI server on your host machine.
6. **Send requests**: Send requests to the host with the appropriate data format and the relevant endpoint.

## Description

The prediction consists of a weighted result of multiple models, including:

- **Text classification model**: Used to predict bigram labels. This model is trained on data gathered from earlier versions of the application and additional data extracted from PDF documents using regular expression matching patterns. It's a multi-label classification task, and the model used is an MLP (Multilayer Perceptron), as there wasn't enough data available for effective use of sequence models. Text augmentation techniques were used to increase the training data.

## More on Text Classification Model

Several methods have been thought of to assist with paraphrasing and augmenting the text:

- **Manually rewrite sentences**: Alter sentence structure while retaining the original meaning.
- **Synonym replacement**: Use a thesaurus or a library like NLTK to replace words with their synonyms.
- **Grammatical restructuring**: Use predefined grammatical rules to restructure sentences.
- **Paraphrasing tools**: Use tools like QuillBot, Paraphraser.io, or APIs from services like TextRazor.
- **Back-translation**: Translate the sentence to another language and back to the original language using services like Google Translate API or MarianMT.
- **Advanced models for paraphrasing**: Use models such as BERT, GPT-3, T5, or Pegasus from Hugging Face Transformers.
- **Text augmentation libraries**: Leverage libraries like `nlpaug` or `textattack` to perform text augmentation techniques.
- **Crowdsourcing platforms**: Use platforms like Amazon Mechanical Turk to have multiple people paraphrase the sentences.

---

Feel free to contribute, open issues, or suggest improvements!


overfitting is clearly visible even with the basic MLP structure (consisting of dense(8192), dropout(.3), dense(4096), dropout(.5), dense(len(multi label vocabulary))

![int 1 gram MLP train data, bigram multihot labels](https://github.com/user-attachments/assets/63bf8ca2-8c37-4249-91b8-e2881d66622c)

![MLP int 2 gram train data, bigram multihot labels](https://github.com/user-attachments/assets/f2f72994-20c1-495a-b477-6d49557c68af)


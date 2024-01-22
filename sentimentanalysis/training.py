import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
import pickle


w_features = set()


# Clean stopwords, hashtags, Twitter handles, and URLs from words
def prepare_words(train):
    dataset = []
    stopwords_set = set(stopwords.words("english"))
    for index, row in train.iterrows():
        if type(row.selected_text) != str:
            continue

        cleaned_words = []
        for word in row.selected_text.split():
            if len(word) < 3:
                continue
            word = word.lower()
            if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT' and word not in stopwords_set:
                cleaned_words.append(word)

        dataset.append((cleaned_words, row.sentiment))

    return dataset


# Feature function that extracts features from word_features and structures them for use by the classifier
def extract_features(document, word_features=None):
    document_words = set(document)
    features = {}

    if not word_features:
        word_features = w_features

    for word in word_features:
        features[word] = (word in document_words)
    return features


def test_accuracy(test, classifier):
    test_negative = test[test['sentiment'] == 'negative']
    test_negative = test_negative['selected_text']
    test_neutral = test[test['sentiment'] == 'neutral']
    test_neutral = test_neutral['selected_text']
    test_positive = test[test['sentiment'] == 'positive']
    test_positive = test_positive['selected_text']

    negative_count = 0
    neutral_count = 0
    positive_count = 0

    for text in test_negative:
        result = classifier.classify(extract_features(text.split()))
        if result == 'negative':
            negative_count += 1

    for text in test_neutral:
        result = classifier.classify(extract_features(text.split()))
        if result == 'neutral':
            neutral_count += 1

    for text in test_positive:
        result = classifier.classify(extract_features(text.split()))
        if result == 'positive':
            positive_count += 1

    print('[Negative]: %s/%s ' % (negative_count, len(test_negative)))
    print('[Neutral]: %s/%s ' % (neutral_count, len(test_neutral)))
    print('[Positive]: %s/%s ' % (positive_count, len(test_positive)))


def training():
    # Read the training data from the CSV and keep only the necessary columns
    data = pd.read_csv('train.csv', encoding='ISO-8859-1')
    data = data[['sentiment','selected_text']]

    # Splitting the dataset into train and test set
    train, test = train_test_split(data, test_size=0.1)

    # Sanitize data
    dataset = prepare_words(train)

    # Get word features and their frequency distributions
    global w_features
    words = []
    for (text, sentiment) in dataset:
        words.extend(text)
    w_features = nltk.FreqDist(words)

    # Save word features frequency to file for use when classifying
    pickle.dump(w_features, open("w_features.pkl", 'wb'))
    w_features = w_features.keys()

    # Structure the training set and train the model
    training_set = nltk.classify.apply_features(extract_features, dataset)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    test_accuracy(test, classifier)

    # Save model to file for use when classifying
    filename = 'preprocess.pkl'
    pickle.dump(classifier, open(filename, 'wb'))


if __name__ == "__main__":
    training()

#reference: https://www.kaggle.com/code/ngyptr/python-nltk-sentiment-analysis/notebook

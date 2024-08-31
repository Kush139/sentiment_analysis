import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import sklearn
from sklearn import svm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from sklearn.metrics import classification_report
from sklearn.utils import class_weight



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



    
def training():
    # Read the training data from the CSV and keep only the necessary columns
    data = pd.read_csv('train.csv', encoding='ISO-8859-1')
    data = data[['sentiment','selected_text']]
    data = data.dropna(subset=['selected_text'])
    dataset = prepare_words(data)
    
    # Convert preprocessed data to DataFrame for easy splitting
    preprocessed_data = pd.DataFrame(dataset, columns=['text', 'sentiment'])
    x = [' '.join(text) for text in preprocessed_data['text']]
    y = preprocessed_data['sentiment']

    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(x_train)
    X_test_tfidf = vectorizer.transform(x_test)

    classifier = svm.SVC(kernel="linear", class_weight='balanced')
    classifier.fit(X_train_tfidf, y_train)

    predictions = classifier.predict(X_test_tfidf)


    report = classification_report(y_test, predictions, target_names=['negative', 'neutral', 'positive'])
    print(report)


    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    training()




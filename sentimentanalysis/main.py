import pickle
import requests
from bs4 import BeautifulSoup
import re
import requests
import nltk
nltk.download('stopwords')
nltk.download('PortStemmer')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize

def preprocess_text(text):
    stopwords_set = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    cleaned_words = []

    for word in words:
        word = word.lower()
        word = stemmer.stem(word)
        if word not in stopwords_set and len(word) > 2:
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words)



def main():
    filename = "svm_model.pkl"
    vectorizer_file = "tfidf_vectorizer.pkl"
    loaded_model = pickle.load(open(filename, "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    choice = int(input("Would you like to enter manual string, or take from a website? Say 1 or 2."))
    print("")
    #input validation
    while choice !=1 and choice !=2:
        choice = int(input("Make sure you choose 1 or 2."))
    if choice == 1:
        text = input("Enter a string ")
        sentiment = analysis(text, loaded_model, vectorizer)
        print("This text seems to be", sentiment)
    else:

        #URL of the website to scrape
        url = input("Enter a URL to scrape:")

        #is it actually a URL?
        while not is_valid_url(url):
            print("Invalid URL. Please enter a valid URL.")
            url = input("Enter a URL to scrape:")

        #Send a request to the website and retrieve the HTML content
        response = requests.get(url)


        #Parse the html contents of the page
        soup = BeautifulSoup(response.content, "html.parser")
        job = int(input("What would you like to scrape from the website? Should I check the sentiment of the whole paragraph, or should I categorize each paragraph by sentiment?"))
        print("")
        #validation
        while job !=1 and job != 2:
            job = int(input("Make sure you choose either 1 or 2."))

        paragraphs = (soup.find_all("p"))

        # Print the text of each paragraph

        # overall sentiment
        if job == 1:
            paragraphs = str(paragraphs)
            sentiment = analysis(paragraphs, loaded_model, vectorizer)
            print("This text seems to be", sentiment)

        # categorize

        else:
            positive = []
            negative = []
            neutral = []
            for p in paragraphs:
                paragraphs = str(paragraphs)
                words = p.text
                sentiment = analysis(words, loaded_model, vectorizer)
                if sentiment == "positive":
                    positive.append(words)
                elif sentiment == "negative":
                    negative.append(words)
                elif sentiment == "neutral":
                    neutral.append(words)
            print("POSITIVE PARAGRAPHS:")
            for i in positive:
                print(i)
            print("NEGATIVE PARAGRAPHS:")
            for i in negative:
                print(i)
            print("NEUTRAL PARAGRAPHS:")
            for i in neutral:
                print(i)


def analysis(text, loaded_model, vectorizer):
    preprocessed_text = preprocess_text(text)
    transformed_text = vectorizer.transform([preprocessed_text])
    sentiment_score = loaded_model.predict(transformed_text)[0]

    return sentiment_score


def is_valid_url(url):
    url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain
        r'localhost|'# localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' #port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_regex.match(url) is not None


if __name__ == "__main__":
    main()

import pickle
import requests
from bs4 import BeautifulSoup
from training import extract_features
import re
import requests

def main():
    filename = "preprocess.pkl"
    loaded_model = pickle.load(open(filename, "rb"))
    loaded_features = pickle.load(open("w_features.pkl", "rb")).keys()
    choice = int(input("Would you like to enter manual string, or take from a website? Say 1 or 2."))
    print("")
    #input validation
    while choice !=1 and choice !=2:
        choice = int(input("Make sure you choose 1 or 2."))
    if choice == 1:
        text = input("Enter a string ")
        sentiment = analysis(text, loaded_model, loaded_features)
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
            sentiment = analysis(paragraphs, loaded_model, loaded_features)
            print("This text seems to be", sentiment)

        # categorize

        else:
            positive = []
            negative = []
            neutral = []
            for p in paragraphs:
                paragraphs = str(paragraphs)
                words = p.text
                sentiment = analysis(words, loaded_model, loaded_features)
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


def analysis(text, loaded_model, loaded_features):
    sentiment_score = loaded_model.classify(extract_features(text.lower().split(), loaded_features))
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

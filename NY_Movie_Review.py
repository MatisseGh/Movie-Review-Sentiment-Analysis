import requests
import datetime
from bs4 import BeautifulSoup


#FIRST PART: GETTING MOVIE REVIEWS

parameters = {'api-key' : 'OUrl7Ja52u2Q9hjGBvqvO83tMnXAA0Wp'}
response = requests.get("https://api.nytimes.com/svc/movies/v2/reviews/search.json", params= parameters)
data = response.json()['results']


#change date strings to datetime objects for sorting
for i in range(len(data)):
    data[i]['date_updated'] = datetime.datetime.strptime(data[i]['date_updated'], '%Y-%m-%d %H:%M:%S')
    if data[i]['opening_date'] is not None and data[i]['opening_date'] != '0000-00-00':
        data[i]['opening_date'] = datetime.datetime.strptime(data[i]['opening_date'], '%Y-%m-%d')
    #Assign a minimum date to null values to order them
    else:
        data[i]['opening_date'] = datetime.datetime.min

#sort by these two keys
data.sort(key = lambda i: (i['opening_date'], i['date_updated']), reverse = True)

#list slice for first 15
first_15 = data[:15]

#function to request the review with beautifulsoup
def get_review(url):
    article = requests.get(url)
    soup = BeautifulSoup(article.content, 'html.parser')
    #this is to remove the last paragraph that isn't part of the review
    for p in soup.find_all("p", {'class':'css-jwz2nf etfikam0'}): 
        p.decompose()
    articleBody = soup.find(attrs={"name":'articleBody'})

    return articleBody.text

#SENTIMENT ANALYSIS VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    snt = analyser.polarity_scores(text)
    return str(snt)

#SENTIMENT ANALYSIS WITH NLTK AND NAIVEBAYES

import nltk
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]


all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]

classifier = nltk.NaiveBayesClassifier.train(featuresets)

#Loop over the first 15 to display the requested info (use range iterator to assign the movie review for use in sentiment analysis)


for i in range(len(first_15)):
    print("Title:", first_15[i]['display_title'])
    #handle opening date null values
    print("Opening date:", datetime.datetime.strftime(first_15[i]['opening_date'], '%Y-%m-%d') if first_15[i]['opening_date'] is not datetime.datetime.min else "No opening date")
    print("Last modified:", datetime.datetime.strftime(first_15[i]['date_updated'], '%Y-%m-%d'))
    print("Author:", first_15[i]['byline'])
    first_15[i]['review'] = get_review(first_15[i]['link']['url'])
    #SENTIMENT VADER
    print("Sentiment score VADER:", get_sentiment_scores(first_15[i]['review']))
    #SENTIMENT NLTK
    print("Sentiment score nltk:")
    dist = classifier.prob_classify(document_features(first_15[i]['review']))
    for label in dist.samples():
        print("\t%s: %f" % (label, dist.prob(label)))

    print("Review:")
    print(first_15[i]['review'])
    print("\n\n", '-'*30, "\n\n", sep='')


classifier.show_most_informative_features(5)

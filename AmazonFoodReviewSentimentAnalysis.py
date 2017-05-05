import re
import numpy
import pandas
import string
import time

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
 
import warnings; warnings.simplefilter('ignore')

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=set(STOPWORDS),
        max_words=400,
        max_font_size=40, 
        scale=3,
        random_state=1
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def expandContractions(text):
    contractionList = {
      "ain't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'll": "he will",
      "he's": "he is",
      "how'll": "how will",
      "how's": "how is",
      "I'd": "I would",
      "I'll": "I will",
      "I'm": "I am",
      "I've": "I have",
      "isn't": "is not",
      "it'll": "it will",
      "it's": "it is",
      "let's": "let us",
      "might've": "might have",
      "must've": "must have",
      "mustn't": "must not",
      "needn't": "need not",
      "never'll":"never will",
      "she'd": "she would",
      "she'll": "she will",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "that'd": "that would",
      "that's": "that is",
      "there's": "there is",
      "they'd": "they would",
      "they'll": "they will",
      "they're": "they are",
      "they've": "they have",
      "wasn't": "was not",
      "we'll": "we will",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "where'd": "where did",
      "where's": "where is",
      "who'll": "who will",
      "who's": "who is",
      "will've": "will have",
      "won't": "will not",
      "would've": "would have",
      "wouldn't": "would not",
      "you'd": "you had",
      "you'll": "you will",
      "you're": "you are",
      "you've": "you have"
    }
    contraction = re.compile('(%s)' % '|'.join(contractionList.keys()))
    def replace(match):
        return contractionList[match.group(0)]
    return contraction.sub(replace, text)


def setstopwords():
    stop_words = stopwords.words('english')
    stop_words.append('URL')
    stop_words.remove('no')
    stop_words.remove('nor')
    stop_words.remove('not')
    return stop_words


def preprocessing(text):
    stop_words = setstopwords()
    stemmer = SnowballStemmer("english")
    try:
        text = re.sub('[0-9]+', ' ', text)
        text = BeautifulSoup(text, "lxml")
        text = text.get_text()
        text = text.lower()
        text = expandContractions(text)
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','URL',text)
        text = re.sub('[\s]+', ' ', text)
        text = re.sub('['+string.punctuation+']', ' ', text)
        text = text.strip('\'"')
        text = " ".join(text.split())
        text = " ".join([word for word in text.split() if word not in stop_words])
        text = " ".join([ stemmer.stem(word) for word in text.split()])
        return text
    except:
        return []


def splitData(data):
    _reviewSummary = []
    _reviewText = []
    _reviewSentiment = []
    for index,row in data.iterrows():
        _sentiment = row['Sentiment']
        _summary = preprocessing(row['Summary'])
        _text = preprocessing(row['Text'])
        #print(_sentiment)
        #print(_summary)
        #print(_text)
        if len(_summary) > 0 and len(_text) > 0:
            _reviewSummary.append(_summary)
            _reviewText.append(_text)
            _reviewSentiment.append(_sentiment)
    return [_reviewSummary,_reviewText,_reviewSentiment]

def sample(data, diff = 0.05, sampling_method = 'O'):

    total_len = len(data)

    total_pos_len = len(data.loc[data['Sentiment'] == 'positive'])
    total_neg_len = len(data.loc[data['Sentiment'] == 'negative'])
    
    to_sample = {}

    dist_list = {'positive':total_pos_len/total_len, 'negative':total_neg_len/total_len}

    max_value = max(dist_list.values())

    for clazz, value in dist_list.items():
        if max_value - value > diff:
            to_add = (max_value - value - diff)
            to_sample[clazz] = int(to_add*total_len)

    to_append = pandas.DataFrame(data.columns)

    for clazz, sample in to_sample.items():
        temp = data.loc[data['Sentiment'] == clazz]
        shuffle(temp)
        if sample < len(temp):
            to_append = pandas.concat([to_append, temp.head(sample)], ignore_index = True)
        else:
            while(sample > 0):
                shuffle(temp)
                to_append = pandas.concat([to_append, temp.head(sample)], ignore_index = True)
                sample = sample - len(temp.head(sample))

    data = pandas.concat([data, to_append], ignore_index=True)

    return data


def build_classifier(classifier_name):
    classifier_pipeline = []
    classifier_pipeline.append(('vectorizer', CountVectorizer()))
    classifier_pipeline.append(('transformer', TfidfTransformer(use_idf = True)))

    if classifier_name == 'LinearSVC':
        classifier_pipeline.append(('clsf', LinearSVC()))
    elif classifier_name == 'MultinomialNB':
        classifier_pipeline.append(('clsf', MultinomialNB()))
    elif classifier_name == 'LogisticRegression':
        classifier_pipeline.append(('clsf', LogisticRegression()))
    elif classifier_name == 'RandomForest':
        classifier_pipeline.append(('clsf', RandomForestClassifier(n_estimators=10)))
    elif classifier_name == 'AdaBoost':
        classifier_pipeline.append(('clsf', AdaBoostClassifier(n_estimators=10)))
   
    return Pipeline(classifier_pipeline)

def classify(data, sentiment, classifiers, k = 10):

    kfold_res = {}

    for classifier_name in classifiers:
        kf = KFold(n_splits = k)
        precision = 0.0
        recall = 0.0
        f1score = 0.0
        accuracy = 0.0

        for train_id, test_id in kf.split(data):
            train_data = [data[i] for i in train_id]
            train_class = [sentiment[i] for i in train_id]
            test_data = [data[i] for i in test_id]
            test_class = [sentiment[i] for i in test_id]
            
            classifier = build_classifier(classifier_name)
            classifier.fit(train_data, train_class)
            
            prediction = classifier.predict(test_data)
            
            precision += precision_score(test_class, prediction, average="macro")
            recall += recall_score(test_class, prediction, average="macro")
            f1score += f1_score(test_class, prediction, average="macro")
            accuracy += accuracy_score(test_class, prediction)

        kfold_res[classifier_name] = {'precision': (precision/k), 'recall': (recall/k), 'f1score':(f1score/k), 'accuracy':(accuracy/k)}

    return kfold_res

def classify_test_reviews(train, train_sentiment, test, test_sentiment, classifiers):

    results = {}

    precision_pos = 0.0
    precision_neg = 0.0
    recall_pos = 0.0
    recall_neg = 0.0
    fscore_pos = 0.0
    fscore_neg = 0.0
    overall_accuracy = 0.0

    results_per_clsf = {}

    for clsf_name in classifiers:

        text_classifier = build_classifier(clsf_name)

        text_classifier.fit(train, train_sentiment)

        prediction = text_classifier.predict(test)

        precision = precision_score(test_sentiment, prediction, labels = ['positive', 'negative'], average=None)
        recall = recall_score(test_sentiment, prediction, labels = ['positive', 'negative'], average=None)
        f1score = f1_score(test_sentiment, prediction, labels = ['positive', 'negative'], average=None)
        accuracy = accuracy_score(test_sentiment, prediction)

        precision_pos += precision[0]
        precision_neg += precision[1]
        recall_pos += recall[0]
        recall_neg += recall[1]
        fscore_pos += f1score[0]
        fscore_neg += f1score[1]
        overall_accuracy += accuracy

        clsf_result = {'precision':precision, 'recall':recall, 'f1score':f1score, 'accuracy':accuracy}

        results_per_clsf[clsf_name] = clsf_result

    print('results per classifier...')  

    print(results_per_clsf)

    print('results of combined classifier...')  

    results = {'accuracy':overall_accuracy/len(classifiers), '1':{'precision':precision_pos/len(classifiers), 'recall':recall_pos/len(classifiers), 'fscore':fscore_pos/len(classifiers)}, '-1':{'precision':precision_neg/len(classifiers), 'recall':recall_neg/len(classifiers), 'fscore':fscore_neg/len(classifiers)}} 

    print(results)

    return results

def format_seconds_to_hhmmss(seconds):
    hours, seconds =  seconds // 3600, seconds % 3600
    minutes, seconds = seconds // 60, seconds % 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def main():
    path = './data/Reviews.csv'
    reviews = pandas.read_csv(path, usecols=['Summary','Text','Score'])
    reviews["Sentiment"] = reviews["Score"].apply(lambda score: "positive" if score > 3 else "negative")
    
    show_wordcloud(reviews[reviews.Sentiment == 'positive']["Summary"])
    show_wordcloud(reviews[reviews.Sentiment == 'negative']["Summary"])
    show_wordcloud(reviews[reviews.Sentiment == 'positive']["Text"])
    show_wordcloud(reviews[reviews.Sentiment == 'negative']["Text"])
    
    print('Review data frame shape : ', reviews.shape)
    print('=============================================================================================')
    print('First 5 records')
    print(reviews.head(5))
    print('=============================================================================================')
    print()

    trainTestSplit = [0.2]
    for split in trainTestSplit:
        train, test = train_test_split(reviews, test_size = split)

        print()
        print('Training Data Before Sampling')
        print('Training Dataset Length = ',len(train))
        trainPos = train[train.Sentiment == 'positive']
        trainNeg = train[train.Sentiment == 'negative']
        print('Positive shape : ', trainPos.shape)
        print('Negative shape : ', trainNeg.shape)
        
        train = sample(train)
        
        print()
        print('Training Data After Sampling')
        print('Training Dataset Length = ',len(train))
        positiveTrainReviews = train[train.Sentiment == 'positive']
        negativeTrainReviews = train[train.Sentiment == 'negative']
        print('Positive shape : ', positiveTrainReviews.shape)
        print('Negative shape : ', negativeTrainReviews.shape)

        print()

        [summary,text,sentiment] = splitData(train)

        classifiers = ['LogisticRegression', 'MultinomialNB', 'LinearSVC', 'RandomForest', 'AdaBoost']

        textSentiment_StartTime = time.time()
        _TextSentiment = classify(text, sentiment, classifiers, k=10)
        textSentiment_EndTime = time.time()
        textSentiment_Time = format_seconds_to_hhmmss(textSentiment_EndTime - textSentiment_StartTime)

        summarySentiment_StartTime = time.time()
        _SummarySentiment = classify(summary, sentiment, classifiers, k=10)
        summarySentiment_EndTime = time.time()
        summarySentiment_Time = format_seconds_to_hhmmss(summarySentiment_EndTime - summarySentiment_StartTime)

        print('Training - kFold results')
        print('---------------------------------------------------------------------------------------------')

        max_acc_score = 0
        max_acc_classifier = ''

        for classifier in classifiers:
            print(classifier)
            print('\t\tPrecision - Summary = ', round(_SummarySentiment[classifier]['precision'], 2), '\t\tPrecision - Text = ', round(_TextSentiment[classifier]['precision'],2))
            print('\t\tRecall - Summary = ', round(_SummarySentiment[classifier]['recall'],2), '\t\tRecall - Text = ', round(_TextSentiment[classifier]['recall'],2))
            print('\t\tF1_score - Summary = ', round(_SummarySentiment[classifier]['f1score'],2), '\t\tF1_score - Text = ', round(_TextSentiment[classifier]['f1score'],2))
            print('\t\tAccuracy - Summary = ', round(_SummarySentiment[classifier]['accuracy'],2), '\t\tAccuracy - Text = ', round(_TextSentiment[classifier]['accuracy'],2))
            print('---------------------------------------------------------------------------------------------')


        print("Execution Time - Training")
        print("Sentiment analysis using Summary : ", summarySentiment_Time)
        print("Sentiment analysis using Text    : ", textSentiment_Time)
        print()

        print('=============================================================================================')

        [test_summary,test_text,test_sentiment] = splitData(test)

        textSentiment_StartTime = time.time()
        print('Classifying based on text')
        classify_test_reviews(text,sentiment, test_text,test_sentiment, classifiers)
        textSentiment_EndTime = time.time()
        textSentiment_Time = format_seconds_to_hhmmss(textSentiment_EndTime - textSentiment_StartTime)

        print()
        summarySentiment_StartTime = time.time()
        print('Classifying based on summary')
        classify_test_reviews(summary,sentiment, test_summary,test_sentiment, classifiers)
        summarySentiment_EndTime = time.time()
        summarySentiment_Time = format_seconds_to_hhmmss(summarySentiment_EndTime - summarySentiment_StartTime)

        print('---------------------------------------------------------------------------------------------')
        print("Execution Time - Testing")
        print("Sentiment analysis using Summary : ", summarySentiment_Time)
        print("Sentiment analysis using Text    : ", textSentiment_Time)
        print()

        print('=============================================================================================')


if __name__ == '__main__':
    main()

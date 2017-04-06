import nltk

from nltk.corpus import stopwords
from bs4 import BeautifulSoup


def preprocess(reviews_df):

	#print(reviews_df['Text'])

	#lower
	reviews_df.Text = reviews_df.Text.str.lower() 
	reviews_df.Summary = reviews_df.Summary.str.lower()

	#add summary to text
	reviews_df['Text'] = reviews_df['Text'] + ' ' +reviews_df['Summary'].map(str)

	#Remove Stopwords
	stop_words = stopwords.words('english')
	stop_words.remove('no');
	stop_words.remove('nor');
	stop_words.remove('not');
	stop_words.remove('won');
	reviews_df['Text'] = reviews_df['Text'].apply(lambda x: ' '.join([item for item in x.split(' ') if item not in stop_words]))

	#Remove HTML Characters
	reviews_df['Text'] = reviews_df['Text'].apply(lambda x: ' '.join(BeautifulSoup(x,"html.parser").findAll(text=True)))

	#remove numbers
	reviews_df['Text'] = reviews_df['Text'].apply(lambda x: ' '.join(word for word in x.split() if not any(char.isdigit() for char in word)))	

	#remove punctuations
	reviews_df['Text'] = reviews_df['Text'].apply(lambda x: ' '.join(word.strip('\'"&@:_!?,.()#+-~=/|*[];$<>') for word in x.split()))

	# Add a class
	reviews_df['Class'] = (reviews_df.Score > 3).astype(int)

	#print(reviews_df['Text'])
	reviews_df.to_csv('test.csv', sep=',')

	return reviews_df
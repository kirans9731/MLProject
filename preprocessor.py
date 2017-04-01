import nltk

from nltk.corpus import stopwords

def preprocess(reviews_df):

	#lower
	reviews_df.Text = reviews_df.Text.str.lower() 

	# Remove Stopwords
	stop_words = stopwords.words('english')
	reviews_df['Text'] = reviews_df['Text'].apply(lambda x: [item for item in x if item not in stop_words])

	# Add a class
	reviews_df['Class'] = (reviews_df.Score > 3).astype(int)

	return reviews_df
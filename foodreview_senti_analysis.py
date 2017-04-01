import pandas
import numpy 
from sklearn.model_selection import train_test_split

from preprocessor import preprocess
from classifier import kfold_cross_validate

def main():
	data_path = 'data/amazon-fine-foods/Reviews.csv'
	reviews_df = pandas.read_csv(data_path)

	#reviews_df = reviews_df.head(1000)

	train, test = train_test_split(reviews_df, test_size = 0.2)

	train = preprocess(train)
	#test = preprocess(test)

	kfold_res = kfold_cross_validate(train, [], k=10)

	print(kfold_res)

if __name__ == '__main__':
	main()
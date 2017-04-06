import pandas
import numpy 
import time
from sklearn.model_selection import train_test_split

from preprocessor import preprocess
from classifier import kfold_cross_validate

def main():

	start_time = time.time()

	data_path = 'data/amazon-fine-foods/Reviews.csv'
	reviews_df = pandas.read_csv(data_path)

	#reviews_df = reviews_df.head(10)

	train, test = train_test_split(reviews_df, test_size = 0.2)

	train = preprocess(train)
	#test = preprocess(test)

	kfold_res = kfold_cross_validate(train, [], k=10)

	print(kfold_res)

	print("Total time take in seconds = %s"% (time.time() - start_time))

if __name__ == '__main__':
	main()
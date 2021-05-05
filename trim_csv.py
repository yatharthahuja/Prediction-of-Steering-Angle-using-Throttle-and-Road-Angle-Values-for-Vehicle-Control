#script to select useful columns only

import pandas as pd 
import numpy as np
import json  
import random

def rectify_trim_csv():

	interim = pd.read_csv("file1.csv")
	trimmed = interim.iloc[:, np.r_[0,4,6]]

	return trimmed

if __name__ == "__main__":
	output_csv_file = rectify_trim_csv()
	print("DONE!!")
	print(output_csv_file)
	output_csv_file.to_csv('test1.csv')
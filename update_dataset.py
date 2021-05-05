import pandas as pd 
import json  
import random

def rectify_random_csv():

	interim = pd.read_csv("master_beta_csv2.csv")
	i=0
	while i<4: 
		factor = 1 + 0.00001*random.randrange(0,200)
		interim.iloc[i, 5] *=factor
		i+=1
	return interim

if __name__ == "__main__":
	output_csv_file = rectify_random_csv()
	print("DONE!!")
	print(output_csv_file)
	output_csv_file.to_csv('master_alpha.csv')
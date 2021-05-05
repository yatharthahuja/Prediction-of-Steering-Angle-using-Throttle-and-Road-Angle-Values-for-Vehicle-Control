# data preprocessing file
# this script imputes missing values and normalises the data 
import pandas as pd

def impute_missing_values(data):

	imputed = data
	columns = imputed.shape[0]
	i = 1
	while(i<columns):
		mean_col = imputed.mean(axis = 0)[i]
		j = 0
		rows = imputed.shape[1]
		while(j<rows):
			if imputed.iloc[ j, i] == 0:
				imputed.iloc[ j, i] = mean_col
			j+=1
		i+=1

	return imputed

def nomralise(data):

	normalised = data 
	columns = normalised.shape[0]
	i = 1
	while(i<columns):
		min_col = normalised[ :, i].min()
		max_col = normalised[ :, i].max()
		rows = normalised.shape[1]
		while(j<rows):
			imputed.iloc[ j, i] = ( imputed.iloc[ j, i] - min_col ) / ( max_col - min_col )
			j+=1
		i+=1

	return nomralised

if __name__ == "__main__":
	data = pd.read_csv("file1.csv")
	output_csv = impute_missing_values(data)
	print("Imputation done!")
	output_csv = nomralise(data)
	print("Normalisation done!")
	pd.DataFrame(output_csv).to_csv("test2.csv")
	print(output_csv)
	print("DONE!!")
# code to convert multiple json files(2018 in number) obtained from Unity simulation into a single csv file
# this csv file, along with road_angle vector obtained from image feature extraction,
# provides the X(parameter value) matrix for regression equations

# importing packages 
import pandas as pd 
import json  

def json_to_csv():

	print("Initialising files...")

	# load master json file using pandas 
	df = pd.read_json('record_0.json', lines = True) 

	print("Initialisation done!")
	print("Starting conversion...")

	i = 1
	while (i <= 4):	
		# load next json file using pandas 
		df2 = pd.read_json('record_'+str(i)+'.json', lines = True) 
		
		# concatenating loaded json file with master json file
		df = pd.concat([df,df2]) 
		print("Completed: "+str(i)+" of 2018...")
		i+=1

	# convert and save dataframe as csv file 
	df.to_csv("master_csv.csv",index=False) 

	# load the resultant csv file 
	result = pd.read_csv("master_csv.csv")

	return result 

if __name__ == "__main__":
	df = pd.read_json('record_0.json', lines = True)
	print(df)
	output_csv_file = json_to_csv()
	pd.DataFrame(output_csv_file).to_csv("json2csv_individual.csv")
	print("DONE!!")
	print(output_csv_file)
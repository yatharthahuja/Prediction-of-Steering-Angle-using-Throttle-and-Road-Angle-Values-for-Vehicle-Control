# script to join master_beta_csv and road_angle to prepare finaldataset file 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

def combine( df1, df2):
	return pd.concat([df1, df2], axis=1, sort=False)

if __name__ == "__main__":
	df1 = pd.read_csv("master_beta_csv2.csv")
	df2 = pd.read_csv("road_angles2.csv")
	output_dataframe = combine(df1, df2)
	output_dataframe.to_csv("prefinal_master_dataset.csv")
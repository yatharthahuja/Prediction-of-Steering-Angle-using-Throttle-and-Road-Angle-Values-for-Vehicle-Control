import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math
import cv2


def gray(image):
    # Convert the image color to grayscale 
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def gaussian_blur(image):
    # Reduce noise from the image 
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def canny_edge_detector(image): 
    # applying canny filter to highlight edges 
    edged = cv2.Canny(image, 50, 150) 
    return edged 

def region_of_interest(image): 
    height = image.shape[0] 
    polygons = np.array([ 
        [(300, height), (1300, height), (550, 250)] 
        ]) 
    mask = np.zeros_like(image) 
      
    # Fill poly-function deals with multiple polygon 
    cv2.fillPoly(mask, polygons, 255)  
      
    # Bitwise operation between canny image and mask image 
    masked_image = cv2.bitwise_and(image, mask)  
    return masked_image 

def aggregrate_slope(lines):

	agg_x1 = (1/2)*(lines[0,0,0]+lines[1,0,0])
	agg_y1 = (1/2)*(lines[0,0,1]+lines[1,0,1])
	agg_x2 = (1/2)*(lines[0,0,2]+lines[1,0,2])
	agg_y2 = (1/2)*(lines[0,0,3]+lines[1,0,3])
	slope = (agg_y1-agg_y2)/(agg_x1-agg_x2)

	return slope

def correction_angle(lines):

	slope = aggregrate_slope(lines)
	angle = (180*math.atan(slope)/math.pi)

	return angle


if __name__ == "__main__":

	# Path of dataset directory 	
	#cap = cv2.VideoCapture("datasets\test2.mp4") 
	i = 0
	n = 1	
	interim_data_array = []
	#while(i<1):#cap.isOpened()): 
		#_, frame = cap.read() 
	while(i<n):
		image=cv2.imread(str(i)+'_cam-image_array_.jpg')
		gray_image = gray(image)
		
		blurred_image = gaussian_blur(gray_image)
		canny_image = canny_edge_detector(blurred_image) 
		#cv2.imshow("",canny_image)
		#cv2.waitKey(0) # waits until a key is pressed
		#cv2.destroyAllWindows() # destroys the window showing image
		lines = cv2.HoughLinesP(canny_image, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5) 
		print(lines)
#		print("Correction Angle: ")
#		angle = correction_angle(lines)
#		print(angle)
#		interim_data_array.append(angle)
#		print(str(i)+"/"+str(n)+" Conversion done...")
		i+=1

#	pd.DataFrame(interim_data_array).to_csv("road_angles.csv")
	print("ALL DONE!!")
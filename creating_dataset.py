import cv2
import numpy as np
import os

parent = 'face_images/'
folders = ['angry', 'happy', 'neutral', 'sad', 'shocked']

training_data = []

for index, folder in enumerate(folders):
	print('now in folder '+folder)
	full_path = os.path.join(parent, folder)
	target = index	#target should be number
	imgs = os.listdir(full_path)

	for img_str in imgs:    # img --> each image name
	   try:
		   img_path = os.path.join(full_path, img_str)
		   img = cv2.imread(img_path)
		   resized_img = cv2.resize(img, (50,50))
		   training_data.append([resized_img, target])
	   except:
		   print('error')

training_data = np.array(training_data)

np.save('dataset_new.npy', training_data)
	   




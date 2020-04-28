import sys, getopt
import os, csv
import cv2
import numpy as np



def main():
	"""
	Function to process the command line arguments. Throws an error if invalid arguments were given
	
	Returns:

	"""
	try:
		# try to read in command line arguments
		# argv[0] is script name, e.g. richardData.py 
		# takes one input command (path)
		# possible input parameters: h and p, where p: means it requires an argument/value
		options, _ = getopt.getopt(sys.argv[1:], "hp:l:", ["help", "path=", "labels="])
	except getopt.GetoptError as err:
		# throw error and terminate script if wrong format
		print(err)
		print('\t{} -p <path_to_data>'.format(sys.argv[0]))
		sys.exit(2)
	
	if len(options) == 0:
		print('Missing arguments!\n\t{} -p <path_to_data>'.format(sys.argv[0]))
		sys.exit(2)


	data_path = ""
	csv_file = ""
	# iterate over command line arguments
	for option, argument in options:
		if option in ("-h", "--help"):  # help
			print('\t{} -p <path_to_data>'.format(sys.argv[0]))
			sys.exit()
		elif option in ("-p", "--path"):  # passed data path
			data_path = argument
		elif option in ("-l", "--labels"):
			if argument.endswith(".csv"):
				csv_file = argument
			else:
				print("Label file has wrong format. Allowed extensions: csv")
				sys.exit()
		else:
			assert False, "unhandeld option"

	return data_path, csv_file


def read_in_images(path):
	"""
	Function to read in all images of the given path
	
	Args:

	Returns:

	"""
	images = []
	# iterate over filenames in given path
	for file_name in os.listdir(path):
		if file_name[0] == ".":
			continue
		# try to read in image with opencv
		img = cv2.imread(os.path.join(path, file_name))
		# if successful, save image to list
		if img is not None:
			images.append(img)

	return np.array(images)


def read_csv(file):
	"""
	Function to read in the labels from the csv file and save the data in XX
	
	Args:

	Returns:

	"""
	# TODO: pandas df vs csv

	return np.array(images)


if __name__== "__main__":
	# process command line args and return given path if args were valid
	path_to_data, label_file = main()

	# read in images from given path
	input_images = read_in_images(path_to_data)
	print('Number of images: {}'.format(len(input_images)))

	# training data configuration. Images will be sheared, zoomed, rescaled, flipped etc to 
	# have a broader variety and hence better training
	# this is helpful for conquering OVERfitting; when encounterin UNDERfitting, reduce preprocessing
	# train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

	# testing generator. To have more variety of images that the network should
	# be tested with. In this case, the images will only be rescaled, but could
	# in theory be more altered  
	# test_generator = ImageDataGenerator(rescale=1. / 255)


	
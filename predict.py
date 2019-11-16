# -*- coding: utf-8 -*-
import os, json, re
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from model import img_class_model

class Predict_Model(object):

	def __init__(self, config_file, weights_file):

		config = self.__load_config(config_file)

		self.model_name = config['model_name']
		self.img_shape = (config['image_height'], config['image_width'], 1)
		self.num_classes = config['num_classes']
		self.lr = config['lr']
		self.weights_file = weights_file

		self.label_names = {
			"0": "NSW",
			"1": "VIC",
			"2": "ACT - Australian Capital Territory",
			"3": "QLD - Queensland",
			"4": "WA",
			"5": "NT - Northern Territory",
			"6": "TAS - Tasmania",
			"7": "SA",
			"8": "New Zealand"
		}

	def __load_config(self, config_path):

		with open(config_path, 'r') as f:
			json_raw = ''.join(f.readlines())
			json_str1 = re.sub(re.compile('(//[\\s\\S]*?\n)'), '', json_raw)
			json_str2 = re.sub(re.compile('(/\*[\\s\\S]*?/)'), '', json_str1)

			return dict(json.loads(json_str2))

	def __load_single_image(self, image_file):

		img = cv2.imread(image_file, 0)
		img = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))
		img = np.array([img_to_array(img)], dtype='float') / 255.0

		return img

	def predict(self, img_file):

		model_obj = img_class_model(image_shape=self.img_shape, classes=self.num_classes, model_name=self.model_name, lr=self.lr)
		model = model_obj.build()

		model.load_weights(self.weights_file)

		img = self.__load_single_image(img_file)

		confidences = model.predict(img).tolist()[0]

		for i, pred in enumerate(confidences):
			print('%s - %-30s\t\t\t%f' % (i, self.label_names[str(i)], pred))


if __name__ == '__main__':

	config_file = r'config.json'
	weights_file = r'checkpoints/weights-ResNet50-08.hdf5'

	# test_img = r'dataset/train/0/637077431134135955-visible.jpg'
	test_img = r'dataset/train/1/637077627915131839-visible.jpg'

	predictModel = Predict_Model(config_file, weights_file)
	predictModel.predict(test_img)

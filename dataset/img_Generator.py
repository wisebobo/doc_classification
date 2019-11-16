# -*- coding: utf-8 -*-
"""
Created on 2019-07-04

@author: benny li
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os, glob, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def img_Generator(inPath, outPath, ratio=1):

	fileLists = glob.glob(inPath + '/*.[jp][pn]g')

	for file in fileLists:

		[dirname, filename] = os.path.split(file)
		out_orig_file = os.path.join(outPath, filename)

		img = Image.open(file)
		img = img.convert("RGB")
		img = img.resize((220, 140), Image.ANTIALIAS)
		img.save(out_orig_file)

		for j in range(ratio):
			single_img_Generator(img=img, outPath=outPath,
				setBlur=False,
				setColor=random.choice([True, False]),
				setBrightness=random.choice([True, False]),
				setSharpness=random.choice([True, False]),
				setContrast=random.choice([True, False])
				)



def single_img_Generator(img, outPath, setBlur=True, setColor=True, setContrast=True, setSharpness=True, setBrightness=False):

	if setBlur:
		img = img.filter(ImageFilter.BLUR)

	if setColor:
		img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.5))

	if setContrast:
		img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.5))

	if setSharpness:
		img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.7, 1.5))

	if setBrightness:
		img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.1))

	data_augment = ImageDataGenerator(
		rotation_range=5,
		# width_shift_range=0.2,
		# height_shift_range=0.2,
		# shear_range=0.2,
		zoom_range=[0.85, 1.15],
		fill_mode='constant',
		# horizontal_flip=True,
		# vertical_flip=True,
		cval=random.randint(235, 255),
		)

	img = np.array(img)
	img = img.reshape((1,) + img.shape)

	im = data_augment.flow(x=img, y=None, batch_size=1, save_to_dir=outPath, save_format='jpg')
	im.next()


def _mkdir(path):

	if os.path.exists(path):
		return True
	else:
		os.makedirs(path)
		return False


if __name__ == '__main__':

	labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]
	train_ratios = [1, 100, 2, 1, 19, 80, 27, 1, 4, 1100, 100, 50, 2, 8, 250, 500, 1000, 1000, 500, 1000, 1000, 500, 1000]
	test_ratios = [1, 20, 2, 1, 13, 70, 14, 1, 3, 200, 50, 50, 2, 70, 200, 200, 200, 200, 200, 200, 200, 200, 200]

	for (label, ratio) in zip(labels, train_ratios):
		inPath = r'train\%s' % label
		outPath = r'train_gen\%s' % label
		_mkdir(outPath)
		img_Generator(inPath, outPath, ratio=ratio)

	for (label, ratio) in zip(labels, test_ratios):
		inPath = r'test\%s' % label
		outPath = r'test_gen\%s' % label
		_mkdir(outPath)
		img_Generator(inPath, outPath, ratio=ratio)

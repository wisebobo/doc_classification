# -*- coding: utf-8 -*-
import json, re, os
from data_loader import DataLoader
from model import img_class_model
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

class Train_Model(object):

	def __init__(self, config_file):

		config = self.__load_config(config_file)

		self.model_name = config['model_name']
		self.checkpoints_path = './checkpoints/' + self.model_name
		self.save_weigths_file_path = "checkpoints/weights-%s-{epoch:02d}.hdf5" % self.model_name
		self.monitor = config['monitor']
		self.lr_reduce_patience = config['lr_reduce_patience']
		self.early_stop_patience = config['early_stop_patience']
		self.batch_size = config['batch_size']
		self.epochs = config['epochs']
		self.initial_epoch = config['initial_epoch']
		self.img_shape = (config['image_height'], config['image_width'], 1)
		self.num_classes = config['num_classes']
		self.lr = config['lr']
		self.weights_file = None

	def __load_config(self, config_path):

		with open(config_path, 'r') as f:
			json_raw = ''.join(f.readlines())
			json_str1 = re.sub(re.compile('(//[\\s\\S]*?\n)'), '', json_raw)
			json_str2 = re.sub(re.compile('(/\*[\\s\\S]*?/)'), '', json_str1)

			return dict(json.loads(json_str2))

	def __mkdir(self, path):
		if not os.path.exists(path):
			return os.mkdir(path)
		return path

	def train(self):

		train_data_loader = DataLoader(images_dir=r'dataset\train_gen', num_classes=self.num_classes, image_shape=self.img_shape)
		test_data_loader = DataLoader(images_dir=r'dataset\test_gen', num_classes=self.num_classes, image_shape=self.img_shape)

		X_train, y_train = train_data_loader.load_data()
		X_test, y_test = test_data_loader.load_data()

		print(X_train.shape)
		print(y_train.shape)
		print(X_test.shape)
		print(y_test.shape)

		model_obj = img_class_model(image_shape=self.img_shape, classes=self.num_classes, model_name=self.model_name, lr=self.lr)
		model = model_obj.build()

		if self.weights_file is not None:
			model.load_weights(self.weights_file)

		tensorboard = TensorBoard(log_dir=self.__mkdir(self.checkpoints_path), histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False)

		lr_reduce = ReduceLROnPlateau(monitor=self.monitor, factor=0.1, patience=self.lr_reduce_patience, verbose=1, mode='auto',  cooldown=0)

		early_stop = EarlyStopping(monitor=self.monitor, min_delta=0, patience=self.early_stop_patience, verbose=1, mode='auto')

		checkpoint = ModelCheckpoint(self.save_weigths_file_path, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

		model.fit(X_train, y_train,
			validation_data=(X_test, y_test),
			epochs=self.epochs,
			batch_size=self.batch_size,
			initial_epoch=self.initial_epoch,
			callbacks=[early_stop, checkpoint, lr_reduce, tensorboard],
			verbose=1)

		# model.fit_generator(
		# 	data_aug.flow(X_train, y_train, batch_size=self.batch_size),
		# 	steps_per_epoch=X_train.shape[0] // self.batch_size,
		# 	validation_data=(X_test, y_test),
		# 	validation_steps=X_test.shape[0] // self.batch_size,
		# 	epochs=self.epochs,
		# 	initial_epoch=self.initial_epoch,
		# 	callbacks=[early_stop, checkpoint, lr_reduce, tensorboard],
		# 	shuffle=True, verbose=1, max_queue_size=1000,
		# )


if __name__ == '__main__':

	config_file = r'config.json'

	trainModel = Train_Model(config_file)
	trainModel.train()

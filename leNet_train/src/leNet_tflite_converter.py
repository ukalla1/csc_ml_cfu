import tensorflow as tf
import keras
from keras.models import Model
import os
import matplotlib as plt


def load_model(path):
	m = tf.keras.models.load_model(path)
	# custom_objects = {"CSC_FC": CustomLayer}
	# with keras.utils.custom_object_scope(custom_objects):
	# 	m = keras.Model.from_config(config)
	return m

def main():
	model_pth = '../bin/models/leNet_trained_test.h5'
	model = load_model(model_pth)
	model.summary()

if __name__ == "__main__":
	main()
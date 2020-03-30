from keras.preprocessing.image import ImageDataGenerator
import numpy
import os

seed = 1

def Generator_train(batch_size=8):
	# data_gen_args = dict(featurewise_center=True,
 #                     featurewise_std_normalization=True,
 #                     rotation_range=90,
 #                     width_shift_range=0.1,
 #                     height_shift_range=0.1,
 #                     zoom_range=0.2)
	data_gen_args = dict(rescale=1./255)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)

	image_generator = image_datagen.flow_from_directory(
	    './dataset/images/train',
	    class_mode=None,
	    target_size=(256,256),
	    batch_size=batch_size,
	    seed=seed)

	mask_generator = mask_datagen.flow_from_directory(
	    './dataset/masks/train',
	    class_mode=None,
	    target_size=(256,256),
	    color_mode="grayscale",
	    batch_size=batch_size,
	    seed=seed)

	# combine generators into one which yields image and masks
	train_generator = zip(image_generator, mask_generator)

	return train_generator

def Generator_val(batch_size=8):
	data_gen_args = dict(rotation_range=0)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)

	image_generator = image_datagen.flow_from_directory(
	    './dataset/masks/validation',
	    class_mode=None,
	    target_size=(256,256),
	    batch_size=batch_size,
	    seed=seed)

	mask_generator = mask_datagen.flow_from_directory(
	    './dataset/masks/validation',
	    class_mode=None,
	    target_size=(256,256),
	    batch_size=batch_size,
	    color_mode="grayscale",
	    seed=seed)

	# combine generators into one which yields image and masks
	validation_generator = zip(image_generator, mask_generator)

	return validation_generator
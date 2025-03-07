import numpy as np
import nibabel as nib
import random
from skimage.exposure import adjust_gamma

def adapt_shape(input_array, target_shape):
	target_shape = target_shape[0:3]
	result = np.zeros(target_shape, dtype=np.float32)
	
	# Calculate padding/cropping amounts for each dimension
	offsets_input = []
	offsets_output = []
	
	for i in range(3):
		if input_array.shape[i] > target_shape[i]:
			# Need to crop
			crop_amount = input_array.shape[i] - target_shape[i]
			start = crop_amount // 2
			offsets_input.append(slice(start, start + target_shape[i]))
			offsets_output.append(slice(0, target_shape[i]))
		else:
			# Need to pad
			pad_amount = target_shape[i] - input_array.shape[i]
			start = pad_amount // 2
			offsets_input.append(slice(0, input_array.shape[i]))
			offsets_output.append(slice(start, start + input_array.shape[i]))
	
	# Copy the data
	result[tuple(offsets_output)] = input_array[tuple(offsets_input)]
	
	return result.astype(np.float32)

def preprocess_image(img_arr,temp_bm):
	"""Clip to [0.01;0.99] and norm to [0;1] (inside brain!)"""
	img_arr = np.clip(img_arr, np.percentile(img_arr[temp_bm != 0],1.),np.percentile(img_arr[temp_bm != 0],99.) )
	img_arr -= img_arr[temp_bm == 1].min()
	img_arr = img_arr / img_arr[temp_bm == 1].max()
	img_arr *= temp_bm
	return img_arr.astype(np.float32)

def yield_batch(ids,input_shape,training=True):
	imgs_input = []
	imgs_target = []
	imgs_gt = []
	for batch_elem in ids:

		batch_input_list = []
		for input_img_name in batch_elem["Input_Images"]:
			input_img = nib.load(input_img_name).get_fdata()
			input_img = adapt_shape(input_img,input_shape[0:3])
			temp_bm = np.zeros(input_shape[0:3])
			temp_bm[input_img != 0] = 1
			batch_input_list.append(preprocess_image(input_img,temp_bm))

		target_img = adapt_shape(nib.load(batch_elem["Target_Image"]).get_fdata(),input_shape[0:3])
		target_img = preprocess_image(target_img,temp_bm)

		if batch_elem["Mask"] != None:
			gt = adapt_shape(nib.load(batch_elem["Mask"]).get_fdata(),input_shape[0:3])
			gt[gt > 0] = 1 #Make sure segmentation is binary
			gt *= temp_bm
		else:
			gt = np.zeros(input_shape[0:3],dtype=np.float32) #For cases w/o segmentation, we only use the loss from the translation task

		if training:
			#Morphology augmentations ... applied to all images
			if random.random() > 0.33:
				axis = random.sample((0,1,2),1)
				for _ in range(len(batch_input_list)):
					batch_input_list[_] = np.flip(batch_input_list[_],axis=axis)
				target_img = np.flip(target_img,axis=axis)
				gt = np.flip(gt,axis=axis) 

			if random.random() > 0.33:
				axis = random.sample((0,1,2),1)
				for _ in range(len(batch_input_list)):
					batch_input_list[_] = np.flip(batch_input_list[_],axis=axis)
				target_img = np.flip(target_img,axis=axis)
				gt = np.flip(gt,axis=axis)

			#Intensity augmentations ... only for images, not segmentation!
			if random.random() > 0.33:
				for _ in range(len(batch_input_list)):
					gamma_ = random.uniform(0.5,1.5)
					batch_input_list[_] = adjust_gamma(batch_input_list[_],gamma=gamma_)
				gamma_ = random.uniform(0.5,1.5)
				target_img = adjust_gamma(target_img,gamma=gamma_)
			
		imgs_input.append(np.stack(batch_input_list,axis=-1))
		imgs_target.append(np.expand_dims(target_img,axis=-1))
		imgs_gt.append(np.expand_dims(gt,axis=-1))

	return np.stack(imgs_input,axis=0), np.stack(imgs_target,axis=0), np.stack(imgs_gt,axis=0)
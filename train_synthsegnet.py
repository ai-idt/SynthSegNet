import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from glob import glob
import numpy as np
import random
import models
import losses
import data_loader

"""
HYPERPARAMETERS
"""
input_shape_real = (192,192,192,2) #Images are concatenated at the final dimension [-1]
input_shape_syn = (192,192,192,1) #For a single output image
gf = 32 #Number of filters in the generator (3D nnUNet)
df = 32 #Number of filters in the discriminator
dropout_rate = 0.0 #dropout rate
batch_size = 1
epochs = 101
seg_lambda = 5. # Gen_Loss: total_gen_loss = (gan_loss + syn_loss) + (seg_lambda * seg_loss) 
model_file_name = "/mnt/Drive4/msseg/generator_model.h5"

"""
FILE IDENTIFICATION - THIS MUST BE ADAPTED TO YOUR USECASE
We require a list of dicts of configuration {"Input_Images": [/path/to/1stinput.nii.gz, /path/to/2ndinput.nii.gz], "GT_Mask": /path/to/mask.nii.gz OR None, "Target_Image": /path/to/target.nii.gz}
Please note that we expect all files to be fully pre-processed, i.e., co-registered and skullstripped

Input_Images: List of input images to load (and stack)
GT_Mask: Either a path to the GT mask (if it exists) or None
Target_Image: Path to the target image (to be synthesized)

The following is an example configuration, where we load flair and t2 as input, seg as mask and t1 as target (i.e., our MS use case)
"""
train_samples_flair = glob("/mnt/Drive4/msseg/**/*_flair.nii.gz",recursive=True)
train_samples = [{"Input_Images": [item, item.replace("_flair.nii.gz","_t2.nii.gz")]} for item in train_samples_flair]
for _ in range(len(train_samples)):
	train_samples[_]["Mask"] = train_samples[_]["Input_Images"][0].replace("_flair.nii.gz","_seg.nii.gz") if os.path.exists(train_samples[_]["Input_Images"][0].replace("_flair.nii.gz","_seg.nii.gz")) else None
	train_samples[_]["Target_Image"] = train_samples[_]["Input_Images"][0].replace("_flair.nii.gz","_t1.nii.gz")

"""
TRAINING FUNCTION
"""
@tf.function
def train_step(src_image, target, gt):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(src_image, training=True) #gen_output is a dict {"out_seg": out_seg, "out_syn": out_syn}

		#Only look at Target_Image in the discriminator
		disc_real_output = discriminator([src_image, target], training=True)
		disc_generated_output = discriminator([src_image, gen_output["out_syn"]], training=True)

		gen_total_loss, gan_loss, syn_loss, seg_loss, dice = losses.generator_loss(disc_generated_output, gen_output["out_syn"], gen_output["out_seg"], target, gt, seg_lambda)
		disc_loss = losses.discriminator_loss(disc_real_output, disc_generated_output)

	generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
	return gen_total_loss, disc_loss, gan_loss, syn_loss, seg_loss, dice, gen_output
	
"""
MAIN
"""
discriminator = models.discriminator_3d(input_shape_real,input_shape_syn,df)
generator = models.nnunet_3d(input_shape_real,gf,dropout_rate)
n_steps = int(epochs * len(train_samples))
generator_optimizer = tf.keras.optimizers.SGD(tf.keras.optimizers.schedules.PolynomialDecay(1e-3,n_steps,1e-5,power=0.9),momentum=0.9,nesterov=True)
discriminator_optimizer = tf.keras.optimizers.SGD(1e-3,momentum=0.9,nesterov=True)

bat_per_epo = int(len(train_samples) / batch_size)
for epoch in range(epochs):
	gen_total_loss_lst, disc_loss_lst, seg_loss_lst, dice_lst = [], [], [], []
	random.shuffle(train_samples)
	print(f'Epoch {epoch+1}/{epochs}')
	bar = tf.keras.utils.Progbar(target=bat_per_epo,stateful_metrics=["Gen_Total_Loss","Disc_Loss","Seg_Loss","Dice"])
	for batch in range(bat_per_epo):
		batch_samples = train_samples[(batch*batch_size):((batch+1)*batch_size)]
		# select a batch of samples
		imgs_input,imgs_target,imgs_gt = data_loader.yield_batch(batch_samples,input_shape_real,training=True)
		gen_total_loss, disc_loss, gan_loss, syn_loss, seg_loss, dice, gen_output = train_step(imgs_input, imgs_target, imgs_gt)
		gen_total_loss_lst.append(gen_total_loss); disc_loss_lst.append(disc_loss)
		if seg_loss > 0:
			seg_loss_lst.append(seg_loss); dice_lst.append(dice)
		bar.update(batch+1,values=[("Gen_Total_Loss", np.mean(gen_total_loss_lst)),("Disc_Loss",np.mean(disc_loss_lst)),("Seg_Loss",np.mean(seg_loss_lst)),("Dice", np.mean(dice_lst))])
	epoch_dict = {"Epoch": epoch+1, "Gen_Total_Loss": np.mean(gen_total_loss_lst), "Disc_Loss": np.mean(disc_loss_lst), "Seg_Loss": np.mean(seg_loss_lst), "Dice": np.mean(dice_lst)}
generator.save(model_file_name)

import tensorflow as tf

mse_loss = tf.keras.losses.MeanSquaredError()

def segmentation_loss(gt, seg_output):
	gt_f = tf.keras.backend.flatten(gt)
	seg_output_f = tf.keras.backend.flatten(seg_output)
	intersection_ = tf.keras.backend.sum(gt_f * seg_output_f)
	sum_ = tf.keras.backend.sum(gt_f) + tf.keras.backend.sum(seg_output_f)
	dice_ = (2. * intersection_ + tf.keras.backend.epsilon()) / (sum_ + tf.keras.backend.epsilon())
	dice_loss_ = 1. - dice_

	return tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt,seg_output)) + dice_loss_, dice_

def generator_loss(disc_generated_output, gen_syn, gen_seg, target, gt, seg_lambda):
	gan_loss = mse_loss(tf.ones_like(disc_generated_output), disc_generated_output) #We want the discriminator to produce "1" for the fake output

	syn_loss = 1. - tf.reduce_mean(tf.image.ssim(target,gen_syn,max_val=1))

	if tf.reduce_mean(gt) > 0.: #Cases with gt segmentation receive supervision (Dice+BCE) from the segmentation aswell
		seg_loss, dice = segmentation_loss(gt, gen_seg)
	else:
		seg_loss, dice = 0., 0.

	total_gen_loss = gan_loss + syn_loss + (seg_lambda * seg_loss) 

	return total_gen_loss, gan_loss, syn_loss, seg_loss, dice

def discriminator_loss(disc_real_output, disc_generated_output):
	real_loss = mse_loss(tf.ones_like(disc_real_output), disc_real_output) #The discriminator should produce "1" for real images

	generated_loss = mse_loss(tf.zeros_like(disc_generated_output), disc_generated_output) #The discriminator should produce "0" for syn images

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss
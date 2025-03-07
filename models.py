import tensorflow as tf

def nnunet_3d(real_input_shape,n_filters=32,dropout_rate=0.0,use_conv_scaling=False):
	"""
	Returns a 3D nnUNet w/ two outputs (seg and syn)
	"""
	
	def conv_3d(in_layer,n_filters,do_dropout=False):
		"""
		Two conv steps per stage with leaky ReLU, instance norm (and dropout)
		Arguments:
			in_layer_: input tensor.
			n_filters: Number of filters
			do_dropout: Boolean whether to do dropout (if dropout_rate > 0)
		"""
		layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
		layer_ = tf.keras.layers.GroupNormalization(groups=-1) (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False) (layer_)
		layer_ = tf.keras.layers.GroupNormalization(groups=-1) (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		if (dropout_rate > 0) and (do_dropout == True):
			layer_ = tf.keras.layers.Dropout(dropout_rate)(layer_)

		return layer_

	def upconv_3d(in_layer,n_filters):
		"""
		Performs a stride 2 transposed convolution
		"""
		if use_conv_scaling:
			layer_ = tf.keras.layers.Conv3DTranspose(n_filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
			layer_ = tf.keras.layers.GroupNormalization(groups=-1) (layer_)
			layer_ = tf.keras.layers.LeakyReLU() (layer_)

		else:
			layer_ = tf.keras.layers.UpSampling3D()(in_layer)

		return layer_

	def downconv_3d(in_layer,n_filters):
		"""
		Performs a stride 2 downconvolution
		"""
		if use_conv_scaling:
			layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
			layer_ = tf.keras.layers.GroupNormalization(groups=-1) (layer_)
			layer_ = tf.keras.layers.LeakyReLU() (layer_)

		else:
			layer_ = tf.keras.layers.MaxPool3D()(in_layer)

		return layer_

	input_layer = tf.keras.layers.Input(real_input_shape, name = "input_layer")

	#Encoder block
	conv_1 = conv_3d(input_layer,n_filters)

	conv_2 = downconv_3d(conv_1,n_filters*2)
	conv_2 = conv_3d(conv_2,n_filters*2)

	conv_3 = downconv_3d(conv_2,n_filters*3)
	conv_3 = conv_3d(conv_3,n_filters*3)

	conv_4 = downconv_3d(conv_3,n_filters*4)
	conv_4 = conv_3d(conv_4,n_filters*4)

	conv_5 = downconv_3d(conv_4,n_filters*5)
	conv_5 = conv_3d(conv_5,n_filters*5)

	#Bottle neck
	bn_conv = downconv_3d(conv_5,n_filters*5)
	bn_conv = conv_3d(bn_conv,n_filters*5)

	#Decoder block
	deconv_5 = upconv_3d(bn_conv,n_filters*5)
	deconv_5 = tf.keras.layers.concatenate([conv_5, deconv_5])
	deconv_5 = conv_3d(deconv_5,n_filters*5,do_dropout=True)

	deconv_4 = upconv_3d(deconv_5,n_filters*4)
	deconv_4 = tf.keras.layers.concatenate([conv_4, deconv_4])
	deconv_4 = conv_3d(deconv_4,n_filters*4,do_dropout=True)

	deconv_3 = upconv_3d(deconv_4,n_filters*3)
	deconv_3 = tf.keras.layers.concatenate([conv_3, deconv_3])
	deconv_3 = conv_3d(deconv_3,n_filters*3,do_dropout=True)

	deconv_2 = upconv_3d(deconv_3,n_filters*2)
	deconv_2 = tf.keras.layers.concatenate([conv_2, deconv_2])
	deconv_2 = conv_3d(deconv_2,n_filters*2,do_dropout=True)

	deconv_1 = upconv_3d(deconv_2,n_filters)
	deconv_1 = tf.keras.layers.concatenate([conv_1, deconv_1])
	deconv_1 = conv_3d(deconv_1,n_filters,do_dropout=True)

	out_seg = tf.keras.layers.Conv3D(1, 1, 1, activation='sigmoid', kernel_initializer='glorot_uniform', padding='same', name = "out_seg") (deconv_1)
	out_syn = tf.keras.layers.Conv3D(1, 1, 1, activation='relu', kernel_initializer='he_uniform', padding='same', name = "out_syn") (deconv_1)

	return tf.keras.Model(input_layer,{"out_seg": out_seg, "out_syn": out_syn})

def discriminator_3d(real_input_shape,syn_input_shape,n_filters=32):
	"""The discriminator (ImageGAN)"""

	def d_layer(layer_input, filters):
		"""Discriminator layer"""

		d_1 = tf.keras.layers.Conv3D(filters, kernel_size=1, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
		d_3 = tf.keras.layers.Conv3D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)

		d = tf.keras.layers.Concatenate()([d_1, d_3])
		d = tf.keras.layers.GroupNormalization(groups=-1) (d)
		d = tf.keras.layers.LeakyReLU()(d)

		d = tf.keras.layers.Conv3D(filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(d)
		d = tf.keras.layers.GroupNormalization(groups=-1) (d)
		d = tf.keras.layers.LeakyReLU()(d)

		return d

	img_real_input = tf.keras.layers.Input(shape=real_input_shape) #This is the input image to the generator
	img_output = tf.keras.layers.Input(shape=syn_input_shape) #This is either real or fake
	# Concatenate image and conditioning image by channels to produce input
	combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_real_input, img_output])

	d1 = d_layer(combined_imgs, n_filters)
	d2 = d_layer(d1, n_filters*2)
	d3 = d_layer(d2, n_filters*3)
	d4 = d_layer(d3, n_filters*4)
	d5 = d_layer(d4, n_filters*5)

	validity = tf.keras.layers.Conv3D(1, kernel_size=1, strides=1, padding='same', activation="linear")(d5)

	return tf.keras.Model([img_real_input, img_output], validity)
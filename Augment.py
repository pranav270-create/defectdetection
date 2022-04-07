import tensorflow as tf
import tensorflow_addons as tfa

#rt_target = 0.6
#tune_kimg = 1 #K images to move
#nimg_ratio = nimg_delta / (tune_kimg * 1000)
#rt = acc['Loss/signs/real']
#strength += nimg_ratio * np.sign(rt - rt_target)

def translate(image, transform):
    # transform of shape [dx, dy]
    return tf.image.translate(image,transform)

def rotate(image, transform):
    return tfa.image.rotate(image, transform)

def crop(image, box, box_indices, crop_size):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = (height, width)
    box = np.random()
    return tf.image.crop_and_resize(image, box, box_indices, crop_size)

def shear_x(image, shear_val, replace):
    return tfa.image.shear_x(image, shear_val, replace)

def shear_y(image, shear_val, replace):
    return tfa.image.shear_y(image, shear_val, replace)

def flip(image, arg):
    if arg == 'horizontal':
        tf.image.flip_left_right(image)
    if arg == 'vertical':
        tf.image.flip_up_down(image)

def transform(image, transform, interpolation = 'nearest', fill_mode = 'nearest'):
    # transform of the form: [a0, a1, a2, b0, b1, b2, c0, c1]
    # (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)
    # k = c0 x + c1 y + 1
    return tfa.image.transform(image, transform, interpolation, fill_mode)

def gate_augment_params(probability, params, disabled_val):
    shape = tf.shape(params)
    cond = (tf.random_uniform(shape[:1], 0, 1) < probability)
    disabled_val = tf.broadcast_to(tf.convert_to_tensor(disabled_val, dtype=params.dtype), shape)
    return tf.where(cond, params, disabled_val)
    
def augment_pipeline(
  
    images,                         # Input images: NCHW, float32, dynamic range [-1,+1].
    labels,                         # Input labels.
    strength         = 1,           # Overall multiplier for augmentation probability; can be a Tensor.
    debug_percentile = None,        # Percentile value for visualizing parameter ranges; None = normal operation.

    # Pixel blitting.
    xflip            = 0,           # Probability multiplier for x-flip.
    rotate90         = 0,           # Probability multiplier for 90 degree rotations.
    xint             = 0,           # Probability multiplier for integer translation.
    xint_max         = 0.125,       # Range of integer translation, relative to image dimensions.

    # General geometric transformations.
    scale            = 0,           # Probability multiplier for isotropic scaling.
    rotate           = 0,           # Probability multiplier for arbitrary rotation.
    aniso            = 0,           # Probability multiplier for anisotropic scaling.
    xfrac            = 0,           # Probability multiplier for fractional translation.
    scale_std        = 0.2,         # Log2 standard deviation of isotropic scaling.
    rotate_max       = 1,           # Range of arbitrary rotation, 1 = full circle.
    aniso_std        = 0.2,         # Log2 standard deviation of anisotropic scaling.
    xfrac_std        = 0.125,       # Standard deviation of frational translation, relative to image dimensions. 
):

    if xflip > 0:
        i = tf.floor(tf.random_uniform([batch], 0, 2))
        i = gate_augment_params(xflip * strength, i, 0)
        if debug_percentile is not None:
            i = tf.floor(tf.broadcast_to(debug_percentile, [batch]) * 2)
        G_inv @= scale_2d_inv(1 - 2 * i, 1)

    # Apply 90 degree rotations with probability (rotate90 * strength).
    if rotate90 > 0:
        i = tf.floor(tf.random_uniform([batch], 0, 4))
        i = gate_augment_params(rotate90 * strength, i, 0)
        if debug_percentile is not None:
            i = tf.floor(tf.broadcast_to(debug_percentile, [batch]) * 4)
        G_inv @= rotate_2d_inv(-np.pi / 2 * i)

    # Apply integer translation with probability (xint * strength).
    if xint > 0:
        t = tf.random_uniform([batch, 2], -xint_max, xint_max)
        t = gate_augment_params(xint * strength, t, 0)
        if debug_percentile is not None:
            t = (tf.broadcast_to(debug_percentile, [batch, 2]) * 2 - 1) * xint_max
        G_inv @= translate_2d_inv(tf.rint(t[:,0] * width), tf.rint(t[:,1] * height))

    # Apply isotropic scaling with probability (scale * strength).
    if scale > 0:
        s = 2 ** tf.random_normal([batch], 0, scale_std)
        s = gate_augment_params(scale * strength, s, 1)
        if debug_percentile is not None:
            s = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * scale_std)
        G_inv @= scale_2d_inv(s, s)

    # Apply pre-rotation with probability p_rot.
    p_rot = 1 - tf.sqrt(tf.cast(tf.maximum(1 - rotate * strength, 0), tf.float32)) # P(pre OR post) = p
    if rotate > 0:
        theta = tf.random_uniform([batch], -np.pi * rotate_max, np.pi * rotate_max)
        theta = gate_augment_params(p_rot, theta, 0)
        if debug_percentile is not None:
            theta = (tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * np.pi * rotate_max
        G_inv @= rotate_2d_inv(-theta) # Before anisotropic scaling.

    # Apply anisotropic scaling with probability (aniso * strength).
    if aniso > 0:
        s = 2 ** tf.random_normal([batch], 0, aniso_std)
        s = gate_augment_params(aniso * strength, s, 1)
        if debug_percentile is not None:
            s = 2 ** (tflib.erfinv(tf.broadcast_to(debug_percentile, [batch]) * 2 - 1) * aniso_std)
        G_inv @= scale_2d_inv(s, 1 / s)

    # Apply post-rotation with probability p_rot.
    if rotate > 0:
        theta = tf.random_uniform([batch], -np.pi * rotate_max, np.pi * rotate_max)
        theta = gate_augment_params(p_rot, theta, 0)
        if debug_percentile is not None:
            theta = tf.zeros([batch])
        G_inv @= rotate_2d_inv(-theta) # After anisotropic scaling.

    # Apply fractional translation with probability (xfrac * strength).
    if xfrac > 0:
        t = tf.random_normal([batch, 2], 0, xfrac_std)
        t = gate_augment_params(xfrac * strength, t, 0)
        if debug_percentile is not None:
            t = tflib.erfinv(tf.broadcast_to(debug_percentile, [batch, 2]) * 2 - 1) * xfrac_std
        G_inv @= translate_2d_inv(t[:,0] * width, t[:,1] * height)

    # ----------------------------------
    # Execute geometric transformations.
    # ----------------------------------

    # Execute if the transform is not identity.
    if G_inv is not I_3:

        # Setup orthogonal lowpass filter.
        Hz = wavelets['sym6']
        Hz = np.asarray(Hz, dtype=np.float32)
        Hz = np.reshape(Hz, [-1, 1, 1]).repeat(channels, axis=1) # [tap, channel, 1]
        Hz_pad = Hz.shape[0] // 4

        # Calculate padding.
        cx = (width - 1) / 2
        cy = (height - 1) / 2
        cp = np.transpose([[-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]]) # [xyz, idx]
        cp = G_inv @ cp[np.newaxis] # [batch, xyz, idx]
        cp = cp[:, :2, :] # [batch, xy, idx]
        m_lo = tf.ceil(tf.reduce_max(-cp, axis=[0,2]) - [cx, cy] + Hz_pad * 2)
        m_hi = tf.ceil(tf.reduce_max( cp, axis=[0,2]) - [cx, cy] + Hz_pad * 2)
        m_lo = tf.clip_by_value(m_lo, [0, 0], [width-1, height-1])
        m_hi = tf.clip_by_value(m_hi, [0, 0], [width-1, height-1])

        # Pad image and adjust origin.
        images = tf.transpose(images, [0, 2, 3, 1]) # NCHW => NHWC
        pad = [[0, 0], [m_lo[1], m_hi[1]], [m_lo[0], m_hi[0]], [0, 0]]
        images = tf.pad(tensor=images, paddings=pad, mode='REFLECT')
        T_in = translate_2d(cx + m_lo[0], cy + m_lo[1])
        T_out = translate_2d_inv(cx + Hz_pad, cy + Hz_pad)
        G_inv = T_in @ G_inv @ T_out

        # Upsample.
        shape = [batch, tf.shape(images)[1] * 2, tf.shape(images)[2] * 2, channels]
        images = tf.nn.depthwise_conv2d_backprop_input(input_sizes=shape, filter=Hz[np.newaxis, :], out_backprop=images, strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        images = tf.nn.depthwise_conv2d_backprop_input(input_sizes=shape, filter=Hz[:, np.newaxis], out_backprop=images, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        G_inv = scale_2d(2, 2) @ G_inv @ scale_2d_inv(2, 2) # Account for the increased resolution.

        # Execute transformation.
        transforms = tf.reshape(G_inv, [-1, 9])[:, :8]
        shape = [(height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
        images = tf.contrib.image.transform(images=images, transforms=transforms, output_shape=shape, interpolation='BILINEAR')

        # Downsample and crop.
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz[np.newaxis,:], strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        images = tf.nn.depthwise_conv2d(input=images, filter=Hz[:,np.newaxis], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        images = images[:, Hz_pad : height + Hz_pad, Hz_pad : width + Hz_pad, :]
        images = tf.transpose(images, [0, 3, 1, 2]) # NHWC => NCHW
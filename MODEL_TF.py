import tensorflow
from tensorflow.keras.layers import concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Model



def EnhancerModelTF(fw,fh):
    comp_tensor = layers.Input(shape=(fh, fw, 3))
    conv_1 = layers.Conv2D(filters=128, kernel_size=[9, 9], padding="same", name='conv_1')(comp_tensor)
    conv_1 = layers.PReLU(name='prelu_1', shared_axes=[1, 2])(conv_1)
    conv_2 = layers.Conv2D(filters=64, kernel_size=[7, 7], padding="same", name='conv_2')(conv_1)
    conv_2 = layers.PReLU(name='prelu_2', shared_axes=[1, 2])(conv_2)
    conv_3 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", name='conv_3')(conv_2)
    conv_3 = layers.PReLU(name='prelu_3', shared_axes=[1, 2])(conv_3)
    conv_4 = layers.Conv2D(filters=32, kernel_size=[1, 1], padding="same", name='conv_4')(conv_3)
    conv_4 = layers.PReLU(name='prelu_4', shared_axes=[1, 2])(conv_4)
    conv_11 = layers.Conv2D(filters=128, kernel_size=[9, 9], padding="same", name='conv_6')(comp_tensor)
    conv_11 = layers.PReLU(name='prelu_6', shared_axes=[1, 2])(conv_11)
    feat_11 = concatenate([conv_1, conv_11], axis=-1)
    conv_22 = layers.Conv2D(filters=64, kernel_size=[7, 7], padding="same", name='conv_7')(feat_11)
    conv_22 = layers.PReLU(name='prelu_7', shared_axes=[1, 2])(conv_22)
    feat_22 = concatenate([conv_2, conv_22], axis=-1)
    conv_33 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", name='conv_8')(feat_22)
    conv_33 = layers.PReLU(name='prelu_8', shared_axes=[1, 2])(conv_33)
    feat_33 = concatenate([conv_3, conv_33], axis=-1)
    conv_44 = layers.Conv2D(filters=32, kernel_size=[1, 1], padding="same", name='conv_9')(feat_33)
    conv_44 = layers.PReLU(name='prelu_9', shared_axes=[1, 2])(conv_44)
    feat_44 = concatenate([conv_4, conv_44], axis=-1)
    conv_10 = layers.Conv2D(filters=3, kernel_size=[5, 5], padding="same", name='conv_out')(feat_44)
    output_tensor = comp_tensor + conv_10
    enhancer = Model(inputs=comp_tensor, outputs=output_tensor)
    return enhancer

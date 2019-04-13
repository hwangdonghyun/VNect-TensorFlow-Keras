# -*- coding: utf-8 -*-

#%%Import libraries
import tensorflow as tf
import tensorflow.keras
import resnet_modules
from tensorflow.keras import layers, models, backend

#%%Functions

number_of_joints = 21
input_image = layers.Input((368, 368, 3))


def slice_tensor(x, k, number_of_joints):
    return x[..., k:k + number_of_joints]


def square_tensor(x):
    return backend.square(x)


def sqrt_tensor(x):
    return backend.sqrt(x)


def locloss(heatmap_gt, y_true, y_pred):
    hmap_gt = heatmap_gt

    locmap_preds = tf.multiply(hmap_gt, y_pred)  #hadamard product
    locmap_gt = tf.multiply(hmap_gt, y_true)

    loss = tf.keras.losses.mean_squared_error(locmap_gt, locmap_preds)
    return loss


#%%Building the VNect model

#ResNet-50 layers of stage 1 to 4f
#Stage1
x = layers.ZeroPadding2D(padding=(3, 3))(input_image)
x = layers.Conv2D(64, (7, 7), 
                  strides=2, 
                  kernel_initializer='he_normal',
                  name='res_1a')(x)
x = layers.BatchNormalization(fused=True,
                              name='bn_1a')(x)
x = layers.Activation('relu')(x)

x = layers.ZeroPadding2D(padding=(1, 1))(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#Stage2
x = resnet_modules.create_conv_block(x,
                                     3, [64, 64, 256],
                                     strides=(1, 1),
                                     stage=2,
                                     block='a')
x = resnet_modules.create_identity_block(x,
                                         3, [64, 64, 256],
                                         stage=2,
                                         block='b')
x = resnet_modules.create_identity_block(x,
                                         3, [64, 64, 256],
                                         stage=2,
                                         block='c')

#Stage3
x = resnet_modules.create_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
x = resnet_modules.create_identity_block(x,
                                         3, [128, 128, 512],
                                         stage=3,
                                         block='b')
x = resnet_modules.create_identity_block(x,
                                         3, [128, 128, 512],
                                         stage=3,
                                         block='c')
x = resnet_modules.create_identity_block(x,
                                         3, [128, 128, 512],
                                         stage=3,
                                         block='d')

#Stage4
x = resnet_modules.create_conv_block(x,
                                     3, [256, 256, 1024],
                                     stage=4,
                                     block='a')  #M
x = resnet_modules.create_identity_block(x,
                                         3, [256, 256, 1024],
                                         stage=4,
                                         block='b')
x = resnet_modules.create_identity_block(x,
                                         3, [256, 256, 1024],
                                         stage=4,
                                         block='c')
x = resnet_modules.create_identity_block(x,
                                         3, [256, 256, 1024],
                                         stage=4,
                                         block='d')
x = resnet_modules.create_identity_block(x,
                                         3, [256, 256, 1024],
                                         stage=4,
                                         block='e')
x = resnet_modules.create_identity_block(x,
                                         3, [256, 256, 1024],
                                         stage=4,
                                         block='f')

#%%VNect Custom Layers
#Stage 5a
res_5a_2 = layers.Conv2D(512,
                         kernel_size=(1, 1),
                         padding='same',
                         name='res_5a_2a')(x)
res_5a_2 = layers.BatchNormalization(fused=True,
                                     name='bn_5a_2a')(res_5a_2)
res_5a_2 = layers.Activation('relu')(res_5a_2)

res_5a_2 = layers.Conv2D(512,
                         kernel_size=(3, 3),
                         padding='same',
                         name='res_5a_2b')(res_5a_2)
res_5a_2 = layers.BatchNormalization(fused=True,
                                     name='bn_5a_2b')(res_5a_2)
res_5a_2 = layers.Activation('relu')(res_5a_2)

res_5a_2 = layers.Conv2D(1024,
                         kernel_size=(1, 1),
                         padding='same',
                         name='res_5a_2c')(res_5a_2)
res_5a_2 = layers.BatchNormalization(fused=True,
                                     name='bn_5a_2c')(res_5a_2)

res_5a_1 = layers.Conv2D(1024, (1, 1), name='res_5a_1')(x)
res_5a_1 = layers.BatchNormalization(fused=True,
                                     name='bn_5a_1')(res_5a_1)

res_5a = layers.add([res_5a_1, res_5a_2])
res_5a = layers.Activation('relu')(res_5a)

#Stage 5b
res_5b = layers.Conv2D(256, kernel_size=(1, 1), name='res_5b_a')(res_5a)
res_5b = layers.BatchNormalization(fused=True,
                                     name='bn_5b_a')(res_5b)
res_5b = layers.Activation('relu')(res_5b)

res_5b = layers.Conv2D(128,
                       kernel_size=(3, 3),
                       padding='same',
                       name='res_5b_b')(res_5b)
res_5b = layers.BatchNormalization(fused=True,
                                     name='bn_5b_b')(res_5b)
res_5b = layers.Activation('relu')(res_5b)

res_5b = layers.Conv2D(256, kernel_size=(1, 1), name='res_5b_c')(res_5b)
res_5b = layers.BatchNormalization(fused=True,
                                     name='bn_5b_c')(res_5b)
res_5b = layers.Activation('relu')(res_5b)

#Stage 5c
res_5c_1 = layers.Conv2DTranspose(128,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding="same",
                                  name='res_5c_1')(res_5b)
res_5c_1 = layers.BatchNormalization(fused=True,
                                     name='bn_5c_1')(res_5c_1)
res_5c_1 = layers.Activation('relu')(res_5c_1)


res_5c_2 = layers.Conv2DTranspose(number_of_joints * 3,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same',
                                  name='res_5c_2')(res_5b)

delta_x = layers.Lambda(slice_tensor,
                        arguments={
                            'k': int(0),
                            'number_of_joints': int(number_of_joints)
                        },
                        output_shape=(92, 92, number_of_joints))(res_5c_2)
delta_y = layers.Lambda(slice_tensor,
                        arguments={
                            'k': int(number_of_joints),
                            'number_of_joints': int(number_of_joints)
                        },
                        output_shape=(92, 92, number_of_joints))(res_5c_2)
delta_z = layers.Lambda(slice_tensor,
                        arguments={
                            'k': int(number_of_joints * 2),
                            'number_of_joints': int(number_of_joints)
                        },
                        output_shape=(92, 92, number_of_joints))(res_5c_2)

delta_x_sqr = layers.Lambda(square_tensor,
                            output_shape=(92, 92, number_of_joints))(delta_x)
delta_y_sqr = layers.Lambda(square_tensor,
                            output_shape=(92, 92, number_of_joints))(delta_y)
delta_z_sqr = layers.Lambda(square_tensor,
                            output_shape=(92, 92, number_of_joints))(delta_z)

bone_length_sqr = layers.Add()([delta_x_sqr, delta_y_sqr, delta_z_sqr])
bone_length = layers.Lambda(sqrt_tensor,
                            output_shape=(92, 92,
                                          number_of_joints))(bone_length_sqr)

res_5c = layers.concatenate([res_5c_2, res_5c_1, bone_length])
res_5c = layers.Conv2D(128, 
                       kernel_size=(3, 3),
                       padding='same',
                       name='res_5c')(res_5c)
res_5c = layers.BatchNormalization(fused=True,
                                   name='bn_5c')(res_5c)
res_5c = layers.Activation('relu')(res_5c)

#Featuremaps
featuremaps = layers.Conv2D(number_of_joints * 4, 
                            kernel_size=(1, 1),
                            name='res_featuremaps')(res_5c)

heatmap_2d = layers.Lambda(slice_tensor,
                           arguments={
                               'k': int(0),
                               'number_of_joints': int(number_of_joints)
                           },
                           output_shape=(92, 92, number_of_joints),
                           name='heatmap_2d')(featuremaps)
loc_heatmap_x = layers.Lambda(slice_tensor,
                              arguments={
                                  'k': int(number_of_joints),
                                  'number_of_joints': int(number_of_joints)
                              },
                              output_shape=(92, 92, number_of_joints),
                              name='loc_heatmap_x')(featuremaps)
loc_heatmap_y = layers.Lambda(slice_tensor,
                              arguments={
                                  'k': int(number_of_joints * 2),
                                  'number_of_joints': int(number_of_joints)
                              },
                              output_shape=(92, 92, number_of_joints),
                              name='loc_heatmap_y')(featuremaps)
loc_heatmap_z = layers.Lambda(slice_tensor,
                              arguments={
                                  'k': int(number_of_joints * 3),
                                  'number_of_joints': int(number_of_joints)
                              },
                              output_shape=(92, 92, number_of_joints),
                              name='loc_heatmap_z')(featuremaps)

m = models.Model(
    inputs=[input_image],
    outputs=[heatmap_2d, loc_heatmap_x, loc_heatmap_y, loc_heatmap_z])
m.summary()

#%%
opt = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

m.compile(optimizer=opt,
          loss={
              'heatmap_2d': 'mean_squared_error',
              'loc_heatmap_x': locloss,
              'loc_heatmap_y': locloss,
              'loc_heatmap_z': locloss
          })

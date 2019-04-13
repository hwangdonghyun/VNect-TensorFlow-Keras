# -*- coding: utf-8 -*-
from tensorflow.keras import layers


def create_conv_block(input_tensor,
                      kernel_size,
                      num_of_filters_list,
                      strides=(2, 2),
                      stage=1,
                      block='a'):
    
    conv_name_str = 'res_' + str(stage) + block + "_"
    bn_name_str = 'bn_' + str(stage) + block + "_"

    x = layers.Conv2D(num_of_filters_list[0], (1, 1),
                      strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2a')(input_tensor)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_of_filters_list[1],
                      kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2b')(x)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_of_filters_list[2], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2c')(x)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2c')(x)

    residual_x = layers.Conv2D(num_of_filters_list[2], (1, 1),
                               strides=strides,
                               kernel_initializer='he_normal',
                               name=conv_name_str + '1')(input_tensor)
    residual_x = layers.BatchNormalization(fused=True,
                                           name=bn_name_str + '1')(residual_x)

    x = layers.add([x, residual_x])
    x = layers.Activation('relu')(x)

    return x


def create_identity_block(input_tensor,
                          kernel_size,
                          num_of_filters_list,
                          stage=1,
                          block='a'):
    
    conv_name_str = 'res_' + str(stage) + block + "_"
    bn_name_str = 'bn_' + str(stage) + block + "_"

    x = layers.Conv2D(num_of_filters_list[0], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2a')(input_tensor)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_of_filters_list[1],
                      kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2b')(x)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_of_filters_list[2], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_str + '2c')(x)
    x = layers.BatchNormalization(fused=True,
                                  name=bn_name_str + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x

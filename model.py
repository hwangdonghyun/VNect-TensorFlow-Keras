# -*- coding: utf-8 -*-

#%%Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models, backend
import resnet_modules

class vnect_model(object):
    def __init__(self, input_size=368, number_of_joints=21):
        self.number_of_joints = number_of_joints
        self.input_size = input_size
        self.eight_of_input_size = int(input_size / 8)

    def slice_tensor(self, x, k):
        return x[..., k:k + self.number_of_joints]

    def square_tensor(self, x):
        return backend.square(x)

    def sqrt_tensor(self, x):
        return backend.sqrt(x)

    
    def loc_heatmap_loss(self, y_true, y_pred):
        hmap_gt = self.heatmap_gt
    
        locmap_preds = tf.multiply(hmap_gt, y_pred)  #hadamard product
        locmap_gt = tf.multiply(hmap_gt, y_true)  #hadamard product
    
        loss = tf.keras.losses.mean_squared_error(locmap_gt, locmap_preds)
        return loss
    
    def custom_loc_heatmap_loss(self, y_true, y_pred):
        heatmap_gt = y_true[...,0:self.number_of_joints]
        loc_heatmap_gt = y_true[...,self.number_of_joints:self.number_of_joints*2]

        loc_heatmap_pred_hadamard = tf.multiply(heatmap_gt, loc_heatmap_gt)  #hadamard product
        loc_heatmap_gt_hadamard = tf.multiply(heatmap_gt, y_true)  #hadamard product
        
        loss = tf.keras.losses.mean_squared_error(loc_heatmap_gt_hadamard, loc_heatmap_pred_hadamard)
        
        return loss
        
        
    def create_network(self):
        #Building the VNect model
        #ResNet-50 layers of stage 1 to 4f
        #Stage1

        input_image = layers.Input((self.input_size, self.input_size, 3))
        x = layers.ZeroPadding2D(padding=(3, 3))(input_image)
        x = layers.Conv2D(64, (7, 7),
                          strides=2,
                          kernel_initializer='he_normal',
                          name='res_1a')(x)
        x = layers.BatchNormalization(fused=True, name='bn_1a')(x)
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
        x = resnet_modules.create_conv_block(x,
                                             3, [128, 128, 512],
                                             stage=3,
                                             block='a')
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

        #VNect Custom Layers
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
        res_5b = layers.Conv2D(256, kernel_size=(1, 1),
                               name='res_5b_a')(res_5a)
        res_5b = layers.BatchNormalization(fused=True, name='bn_5b_a')(res_5b)
        res_5b = layers.Activation('relu')(res_5b)

        res_5b = layers.Conv2D(128,
                               kernel_size=(3, 3),
                               padding='same',
                               name='res_5b_b')(res_5b)
        res_5b = layers.BatchNormalization(fused=True, name='bn_5b_b')(res_5b)
        res_5b = layers.Activation('relu')(res_5b)

        res_5b = layers.Conv2D(256, kernel_size=(1, 1),
                               name='res_5b_c')(res_5b)
        res_5b = layers.BatchNormalization(fused=True, name='bn_5b_c')(res_5b)
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

        res_5c_2 = layers.Conv2DTranspose(self.number_of_joints * 3,
                                          kernel_size=(4, 4),
                                          strides=(2, 2),
                                          padding='same',
                                          name='res_5c_2')(res_5b)

        res_5c_2_sqr = layers.Lambda(
            self.square_tensor,
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints * 3))(res_5c_2)
        delta_x_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(0)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(res_5c_2_sqr)
        delta_y_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(res_5c_2_sqr)
        delta_z_sqr = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints * 2)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(res_5c_2_sqr)
        '''
        delta_x_sqr = layers.Lambda(square_tensor,
                                output_shape=(92, 92, self.number_of_joints))(delta_x)
        delta_y_sqr = layers.Lambda(square_tensor,
                                output_shape=(92, 92, self.number_of_joints))(delta_y)
        delta_z_sqr = layers.Lambda(square_tensor,
                                output_shape=(92, 92, self.number_of_joints))(delta_z)
        '''
        bone_length_sqr = layers.Add()([delta_x_sqr, delta_y_sqr, delta_z_sqr])
        bone_length = layers.Lambda(
            self.sqrt_tensor,
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints))(bone_length_sqr)

        res_5c = layers.concatenate([res_5c_2, res_5c_1, bone_length])
        res_5c = layers.Conv2D(128,
                               kernel_size=(3, 3),
                               padding='same',
                               name='res_5c')(res_5c)
        res_5c = layers.BatchNormalization(fused=True, name='bn_5c')(res_5c)
        res_5c = layers.Activation('relu')(res_5c)

        #Featuremaps
        featuremaps = layers.Conv2D(self.number_of_joints * 4,
                                    kernel_size=(1, 1),
                                    name='res_featuremaps')(res_5c)

        heatmap_2d = layers.Lambda(self.slice_tensor,
                                   arguments={'k': int(0)},
                                   output_shape=(self.eight_of_input_size,
                                                 self.eight_of_input_size,
                                                 self.number_of_joints),
                                   name='heatmap_2d')(featuremaps)
        loc_heatmap_x = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints),
            name='loc_heatmap_x')(featuremaps)
        loc_heatmap_y = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints * 2)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints),
            name='loc_heatmap_y')(featuremaps)
        loc_heatmap_z = layers.Lambda(
            self.slice_tensor,
            arguments={'k': int(self.number_of_joints * 3)},
            output_shape=(self.eight_of_input_size, self.eight_of_input_size,
                          self.number_of_joints),
            name='loc_heatmap_z')(featuremaps)

        m = models.Model(
            inputs=[input_image],
            outputs=[heatmap_2d, loc_heatmap_x, loc_heatmap_y, loc_heatmap_z])
        m.summary()
    
        return m

#%%
        
v = vnect_model(368, 21)
model = v.create_network()

ada = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ada, loss={'loc_heatmap_x':v.custom_loc_heatmap_loss, 
                                   'loc_heatmap_y':v.custom_loc_heatmap_loss, 
                                   'loc_heatmap_z':v.custom_loc_heatmap_loss})
    
        

# VNect-TensorFlow-Keras

This is an unofficial implementation of a VNect model with TF-Keras referring to **"VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera"** paper in SIGGRAPH'17.  [[site](http://gvv.mpi-inf.mpg.de/projects/VNect/)]
 
## Network Structure

![VNect Network Image](https://www.researchgate.net/profile/Dan_Casas/publication/316679944/figure/fig2/AS:490492136824833@1493953846900/Network-Structure-The-structure-above-is-preceded-by-ResNet50-100-till-level-4-We-use_W640.jpg)


The network structure of VNect is shown in the figure above.  
The graph of the network model based on this structure is shown in the [link](./image/model.png).

## Files
[model.py](./model.py): The main implementation of the VNect network.  
[resnet_modules.py](./resnet_modules.py): The implementation of the ResNet[[site](https://arxiv.org/abs/1512.03385)] blocks used for the VNect implementation.

## Training the Network
The current implementation provides a loss function for 3D location maps.  
It has no specific code for training the network and will be updated later.

If you want to train this network, you need to take the following steps:  
1. Train the 2D heatmap estimation.
	* The original VNect paper describes that they used multiple stage training.
	* However, the paper does not specifically describe which modules are used for 2D heatmap estimation; therefore, it is recommended to first modify the network by reducing the number of output filters (e.g. top layer or 'res_5c_1' layer) and then training the network with 2D pose datasets such as MPII and LSP. 
	* Use the mean square error as a loss function.
2. Fine-tune the network with 3D dataset.
	* Load the pre-trained network and recover layers for location heatmaps then fine-tune it with MPI-INF3DHP dataset.
	* My implementation provides the custom loss function for training location heatmaps.   
	Because tf.keras doesn't support multiple arguments for a loss function, we implemented it with a temporary way, and you should consider the following things when you design the data generator.
		* The data generator should provide the ground truth of the locationmap as well as the ground truth of the 2D heatmap.
		* Example
			+ Shape of the ground truth: [42,42,21]
			+ The ground truth output of the data generator should be [42,42,42]  
			  First 21 channels are for 2D heatmaps and remainings are for location heatmaps.
			+ The data generator have to provide this type of data for each location heatmaps x, y, z.

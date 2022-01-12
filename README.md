 <h1 align="left">Spectral convolution with guided filter in denoising image</h1>
<h2 align="center">  
 
 ## Goal
Using spectral convolution to gain the pixel-to-pixel relationship in images and make this information as a guided filter in image processing e.g., denoising image. 
 
  ## Introduction
  
Normal convolution could be deal with data with regular coordination (e.g., Cartesian coordinate). However, not all input could be presented in regular coordination, such as social network relationship, multi-sensor audio processing. Therefore, spectral convolution treated the input as a graph, with each representation as the vertice in graph, and 

<p align="center">
<img src="https://github.com/ychuang1234/Spectral-convolution-with-guided-filter-in-denoising-image/blob/412cabe205045339f7b8a942a5eed8f43c078790/spectral_conv.JPG" width="80%"></p>  
  
  ## Description
I implemented Baysiean optimization algorithm with gaussian model to sample the possible combinations of hyperparamters in KNN model. The dataset was created randomly with 5 cluster with 2D feature (a.k.a number of features is two), which were not disclosed in the real scenario in training process. I randomly sampled 50 combinations of hyperparameters to make the gaussian model efficienly simulate the relationship between hypermeters and overall performance of the KNN model. Through simulation with Baysiean optimization (maximizing the posterior probility), instead of training to get the real data of the model performation, which is time-comsuming.
<p align="center">
<img src="https://github.com/ychuang1234/Spectral-convolution-with-guided-filter-in-denoising-image/blob/8b44a91e55091fa1102a16e8144d7e59370cee49/procedure.JPG" width="80%"></p>


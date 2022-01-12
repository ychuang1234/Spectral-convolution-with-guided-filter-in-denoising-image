 <h1 align="left">Spectral convolution with guided filter in denoising image</h1>
<h2 align="center">  
 
 ## Goal
Using **spectral convolution** to gain the **pixel-to-pixel relationship in images** and make this information as a guided filter in image processing e.g., denoising image. 
 
  ## Introduction
  
Normal convolution could be deal with data with regular coordination (e.g., Cartesian coordinate). However, not all input could be presented in regular coordination, such as **social network relationship**, **multi-sensor audio processing**. Therefore, spectral convolution **treated the input as a graph**, with each representation as the vertice in graph, and the connection (weights) could be defined by the data scientist to **illustrate the similarity between two vertices**, making the possibility to process data in **irregular structure**. 

<p align="center">
<img src="https://github.com/ychuang1234/Spectral-convolution-with-guided-filter-in-denoising-image/blob/412cabe205045339f7b8a942a5eed8f43c078790/spectral_conv.JPG" width="80%"></p>  
  
  ## Experiment procedure and result

  In order to show the if intra-pixels information as guided filter could benefit the output of the denoising image, I make two images, which are from spectral convolution and original image, as the guided filter in denoising operation. However, in order to reduce the processing time in laplacian matrix contruction, I firstly down sampling the images to cut down the number of pixels in images. After being spectrally convolved (actually it is only a dot production in frequency domain), I up-sampling the output images to the original size as the guided filter in denoising image. The result (test_smoothed v.s test_L_smoothed) could show that the output with intra-pixels information could preserve more edge details and show more color contrast compared with output without extra information.
 <p align="center">
<img src="https://github.com/ychuang1234/Spectral-convolution-with-guided-filter-in-denoising-image/blob/8b44a91e55091fa1102a16e8144d7e59370cee49/procedure.JPG" width="80%"></p>
 


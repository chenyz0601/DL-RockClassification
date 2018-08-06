# DL-RockClassification
## Problem description
Deep learning for rock classification, which is similar with semantic segmentation in computer vision.<br>
Input a multi-band remote sensing image, the purpose is to classify each pixel in the image into different classes of rocks.<br>
![An example](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/example.png)<br>
## Model: Conditional GAN
The model has too parts: Generator(a segmentation network) and Discriminator(an adversarial network)<br>
#### Generator
Generator follows the design of [U-net](https://arxiv.org/pdf/1505.04597.pdf).<br>
![the structure of U--net](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/G.png)<br>
In this model, using a list to control the number of filters of each encoder or decoder. And the last layer is using softmax for multi-class classification.<br>
Using AlphaDropout, and dropout rate is lower as network goes deeper, using Conv2DTranspose to do the deconvolution.<br>
<br>
#### Discriminator
Discriminator takes the multi-band remote sensing image and corresponding label as input, outputs a number in range [0,1], which indicates the probability of that the input label is not produced by the Generator.<br> 
![The structure of adversarial network:](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/D.png)<br>
#### Phase: Train Discriminator
To train discriminator, we minimize the loss:<br>
$$loss_d(\theta_d) = \sum_{n=1}^{N}l_{bce}(D(X_n, Y_n), 1) + l_{bce}(D(X_n, G(X_n)), 0),$$
where $l_{bce}$ is the binary crossentropy, $D(x,y)$ is the discriminator, $G(x)$ is the generator, $\theta_d$ means the parameters of dicriminator. In this training phase, we try to trian discriminator to recognize the ground truth label from prediction of generator. And the parameters of generator $\theta_g$ are fixed during this phase.<br>
#### Phase: Train Generator
To train the generator, we minimize the loss:<br>
$$loss_g(\theta_g) = \sum_{n=1}^{N}l_{cce}(G(X_n), Y_n) + \lambda l_{bce}(D(X_n, G(X_n)), 1),$$
where $l_{cce}$ is the categorical crossentropy, $\lambda$ is the weight to balance two part of this loss, $\theta_g$ means the parameters of generator. In this training phase, we try to optimize generator to predict the ground truth distribution and also improve the prediction to fool the discriminator. And the parameters of discriminator $\theta_d$ are fixed during this phase.<br>
## Data preprocessing
use arcpy jupyter notebook API to open pre-processing.ipynb<br>
all images are re-sampled into 10m spatial resolution.<br>
for simplicity, the large remote sensing images are cropped into small tiles with size of 256X256.<br>
each band is normalized by (-max)/(max-min)<br>
#### ASTER
As an advanced multispectral sensor launched onboard Terra spacecraft in December 1999, ASTER covers a broad ragne of spectral region with 14 spetral bands, including three VNIR bands with 15m spatial resoltion, six SWIR bands with 30m spatial resolution, and fice TIR bands with 90m spatial resolution.<br>
Time: 01/04/2014 - 01/06/2014<br>
Cloud coverage: 0.0%-0.0%<br>
|Band|Central Wavelength(nm)|Spatial Resolution(m)|
|----|----------------------|---------------------|
|1   |0.5560                |15                   |
|2   |0.6610                |15                   |
|3N  |0.8070                |15                   |
|3B  |0.8070                |15                   |
|4   |1.6560                |30                   |
|5   |2.1670                |30                   |
|6   |2.2090                |30                   |
|7   |2.2620                |30                   |
|8   |2.3360                |30                   |
|9   |2.4000                |30                   |
|10  |8.2910                |90                   |
|11  |8.6340                |90                   |
|12  |9.0750                |90                   |
|13  |10.6570               |90                   |
|14  |11.3180               |90                   |
#### Sentinel-2A
The Sentinel-2A image contains 13 spectral bands in the VNIR and SWIR spectral range, with four bands at 10m, six bands at 20m, and three atmospheric correction bands at 60m spatial resolution. The cloud free image was automatically atmospherically corrected using the Sentinel Application Platform software package provided by ESA.<br>
Time: 01/04/2018 - 01/05/2018<br>
Cloud coverage: 0.0%-0.0%<br>
|Band|Central Wavelength(nm)|Spatial Resolution(m)|
|----|----------------------|---------------------|
|1   |0.4430                |60                   |
|2   |0.4900                |10                   |
|3   |0.5600                |10                   |
|4   |0.6650                |10                   |
|5   |0.7050                |10                   |
|6   |0.7400                |20                   |
|7   |0.7830                |20                   |
|8   |0.8420                |10                   |
|8A  |0.8650                |20                   |
|9   |0.9450                |60                   |
|10  |1.3750                |60                   |
|11  |1.6100                |60                   |
|12  |2.1900                |20                   |
#### Geophysical data
#### composite all useful bands into a multi-band raster
| band  | mean |
|-------|------|
| 0     | Blue |
| 1     | Green |
| 2     | Red |
| 3     | VRE |
| 4     | VRE |
| 5     | VRE |
| 6     | NIR |
| 7     | SWIR |
| 8     | SWIR |
| 9     | VRE |
| 10    | K |
| 11    | Kpcent |
| 12    | TH |
| 13    | U |
| 14    | U2 |
| 15    | magnetic |
| 16    | U over K |
| 17    | U over TH |
| 18    | K over TH |
| 19    | K over U |
| 20    | U2 over TH |
| 21    | TH over U |
| 22    | TH over K |

#### layer number and rock type
| Layer | Rock |
|-------|------|
| 0     | Vegetation|
| 1     | Unkown Rocks|
| 2     | Carbonate_sediment|
| 3     | Dolerite|
| 4     | Feldspathic_sediment|
| 5     | Felsic_volcanic|
| 6     | Gneiss|
| 7     | Granite|
| 8     | Mafic_volcanic|
| 9     | Quartz_sediment|

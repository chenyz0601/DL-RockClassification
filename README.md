# DL-RockClassification
## Problem description
Deep learning for rock classification, which is similar with semantic segmentation in computer vision.<br>
Input a multi-band remote sensing image, the purpose is to classify each pixel in the image into different classes of rocks.<br>
![An example](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/example.png)<br>
## Model: Conditional GAN
The model has too parts: Generator(a segmentation network) and Discriminator(an adversarial network)<br>
#### Generator
Generator follows the design of [U-net](https://arxiv.org/pdf/1505.04597.pdf).<br>
![the structure of U--net]()<br>
In this model, using a list to control the number of filters of each encoder or decoder. And the last layer is using softmax for multi-class classification.<br>
Using AlphaDropout, and dropout rate is lower as network goes deeper, using Conv2DTranspose to do the deconvolution.<br>
<br>
#### Discriminator
Adversarial net takes the multi-band remote sensing image and corresponding label as input, outputs a number in range [0,1], which indicates the probability of that label is not produced by the segmentation net.<br> 
![The structure of adversarial network:](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/AdversarialNet.png)<br>
#### Train phase: adversarial network
To train the adversarial neural network, we minimize the loss:<br>
$$loss_a(\theta_a) = \sum_{n=1}^{N}l_{bce}(a(X_n, Y_n), 1) + l_{bce}(a(X_n, s(X_n)), 0),$$
where $l_{bce}$ is the binary crossentropy, $a(x,y)$ is the adversarial network, $s(x)$ is the segmentation network, $\theta_a$ means the parameters of adversarial network. In this training phase, we try to optimize adversarial network to discriminate the ground truth and prediction from segmentation network. And the parameters of segmentation network $\theta_s$ are fixed during this phase.<br>
#### Training phase: segmentation network
To train the segmentation neural network, we minimize the loss:<br>
$$loss_s(\theta_s) = \sum_{n=1}^{N}l_{cce}(s(X_n), Y_n) + \lambda l_{bce}(a(X_n, s(X_n)), 1),$$
where $l_{cce}$ is the categorical crossentropy, $\lambda$ is the weight to balance two part of this loss, $\theta_s$ means the parameters of segmentation network. In this training phase, we try to optimize segmentation network to predict the ground truth distribution and also improve the prediction to fool the adversarial network. And the parameters of adversarial network $\theta_a$ are fixed during this phase.<br>
## Data preprocessing
use arcpy jupyter notebook API to open pre-processing.ipynb<br>
all images are re-sampled into 10m spatial resolution.<br>
for simplicity, the large remote sensing images are cropped into small tiles with size of 256X256.<br>
each band is normalized by (-max)/(max-min)<br>
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

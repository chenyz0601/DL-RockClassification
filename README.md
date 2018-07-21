# DL-RockClassification
## Problem description
Deep learning for rock classification, which is similar with semantic segmentation in computer vision.<br>
Input a multi-band remote sensing image, the purpose is to classify each pixel in the image into different classes of rocks.<br>
![An example](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/example.png)<br>
## Model: Segmentation net + Adversarial net
The model has too parts: Segmentation net and Adversarial net, follows the description in [Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408.pdf)<br> 
Segmentation net is based on [U-net](https://arxiv.org/pdf/1505.04597.pdf).<br>
![the structure of U--net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)<br>
In this model, using a list [64,64,64,64] to control the number of filters of each encoder or decoder. And the last layer is using softmax for multi-class classification.<br>
Using AlphaDropout, and dropout rate is lower as network goes deeper, using Conv2DTranspose to do the deconvolution.<br>
<br>
Adversarial net takes the multi-band remote sensing image and corresponding label as input, outputs a number in range [0,1], which indicates the probability of that label is not produced by the segmentation net.<br> 
![The structure of adversarial network:](https://github.com/chenyz0601/DL-RockClassification/blob/master/img/AdversarialNet.png)<br>

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

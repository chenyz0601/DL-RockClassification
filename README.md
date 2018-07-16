# DL-RockClassification
deep learning for rock classification, which is similar with semantic segmentation in computer vision<br>
This model has too parts: Segmentation net and Adversarial net, as described in [Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408.pdf)<br> 
Segmentation net is based on [U-net](https://arxiv.org/pdf/1505.04597.pdf)<br>
![the structure of U--net](https://www.google.de/search?q=u-net&rlz=1C1GCEA_enDE791DE791&source=lnms&tbm=isch&sa=X&ved=0ahUKEwi7u9fdsKPcAhVGL1AKHSEuAKMQ_AUICigB&biw=1689&bih=1031#imgrc=K-wr6kKnRJ4HPM:)<br>
Adversarial net takes the multi-band remote sensing image and corresponding label as input, outputs a number in range [0,1], which indicates the probability of that label is not produced by the segmentation net.<br> 
use arcpy jupyter notebook API to open pre-processing.ipynb<br>
each band is normalized by (-max)/(max-min)<br>
#### composite all useful bands into a multiband raster
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
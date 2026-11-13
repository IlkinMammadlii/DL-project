# Attribution and licenses

Project creators: **Sabuhi Aghayev and Ilkin Mammadli**.

## Architecture and implementation

PointNet++ is the architecture introduced by Qi, Yi, Su and Guibas, NeurIPS 2017.
This project adapts an existing model family and does not claim its invention.
Original utility code is retained under `source/pointnet2/vendor/` with the MIT
notice and pinned source commit `eb64fe0b4c24055559cea26299cb485dcb43d8dd`.
Upstream: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The single-file script also contains the upstream MIT notice.

## Data and figures

Dataset: SemanticKITTI, Behley et al., ICCV 2019; KITTI, Geiger et al., CVPR 2012.
Official data and terms: https://www.semantic-kitti.org/dataset.html
Mirror used: https://huggingface.co/datasets/larryshaw0079/SemanticKITTI-SP

KITTI/SemanticKITTI data and derived camera/point-cloud figures retain their
upstream attribution, noncommercial and share-alike restrictions. They are not
relicensed under this repository's code MIT license. No raw dataset is included.
The semantic mapping/API MIT notice is retained in
`source/pointnet2/SEMANTICKITTI-API-LICENSE.txt`.

Results figures contain real measurements and model predictions. Semantic
overlays label projected LiDAR points rather than every camera pixel.

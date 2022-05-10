# RSSGL
RSSGL: Statistical Loss Regularized 3D ConvLSTM for Hyperspectral Image Classification

Code download link: [RSSGL code](https://github.com/swiftest/RSSGL).

Here is the bibliography info:
> @article{wang2022RSSGL,  
> &emsp; title={RSSGL: Statistical Loss Regularized 3D ConvLSTM for Hyperspectral Image Classification},  
> &emsp; author={Wang, Liguo and Wang, Heng and Wang, Lifeng and Wang, Xiaoyi and Shi, Yao and Cui, Ying},  
> &emsp; journal={IEEE Transactions on Geoscience and Remote Sensing},  
> &emsp; year={2022},  
> &emsp; DOI (identifier)={10.1109/TGRS.2022.3174305},  
> &emsp; publisher={IEEE}  
> }


# Steps:
- 1. Run 'bash setup_script.sh' to download the data sets. Then, put the data sets and the ground truth into the corresponding folders.
- 2. Unizp the 'simplecv.zip' into your PYTHONPATH, or move the unzipped module to the 'site-packages' path.
- 3. Run 'bash ./scripts/....sh' to reproduce the experiments presented in the Paper.


# Descriptions

![](figure/fig1.png)

Fig1. Overall architecture of the proposed RSSGL. Given a full hyperspectral dataset of size $${H \times W \times B}$$, where $B$ indicates the number of spectral bands, the unified standardized input feature map is passed through 3D ConvLSTM to learn the short-range and long-range cross-channel dependencies and global spatial context features. Then, abundant spectral-spatial features are extracted through GJAM and group normalization (GN) is used to correct the inaccurate batch statistics estimation. Finally, the softmax layer is used for classification, and cross-entropy combined with statistical loss are used for error backward propagation.


![](figure/fig2.png)

Fig2. The architecture of the 3D ConvLSTM.


![](figure/fig3.png)

Fig3. The architecture of the 3D ConvLSTMCell.



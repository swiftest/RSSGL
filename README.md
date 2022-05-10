# RSSGL
RSSGL: Statistical Loss Regularized 3D ConvLSTM for Hyperspectral Image Classification

Steps:
1. Please run 'bash setup_script.sh' to download the data sets. Then, put the data sets and the ground truth into the corresponding folders.
2. Unizp the 'simplecv.zip' into your PYTHONPATH, or move the unzipped module to the 'site-packages' path.
3. Run 'bash ./scripts/....sh' to reproduce the experiments presented in the Paper.


# Description

![](figure/fig1.png)

Fig1. Overall architecture of the proposed RSSGL. Given a full hyperspectral dataset of size ${H \times W \times B}$, where $B$ indicates the number of spectral bands, the unified standardized input feature map is passed through 3D ConvLSTM to learn the short-range and long-range cross-channel dependencies and global spatial context features. Then, abundant spectral-spatial features are extracted through GJAM and group normalization (GN) is used to correct the inaccurate batch statistics estimation. Finally, the softmax layer is used for classification, and cross-entropy combined with statistical loss are used for error backward propagation.


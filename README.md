# Variational-Scaled-Dirichlet-Mixture-Model

This is the implementation of a Variational Scaled Dirichlet Mixture Model focusing on data clustering tasks on texture and object images. The results are published in IEEE ISIE 2019 paper: "Data Clustering using Variational Learning of Finite Scaled Dirichlet Mixture Models" at https://ieeexplore.ieee.org/document/8781334

The model in model.py is built from scratch while GMM and VGMM in gmm.py are from scikit-learn library.

The preprocessing steps are for preprocessing images using co-occurrence matrix from Vistex dataset (MIT Media Lab). Each image are split into 8 smaller images to increase size of the dataset.

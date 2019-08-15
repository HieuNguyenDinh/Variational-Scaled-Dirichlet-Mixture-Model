import numpy as np
from sklearn import mixture, preprocessing
from scipy.cluster.vq import kmeans, kmeans2, vq
import pandas as pd
import timeit

import ground_truth

def GMM_VGMM(dat, clusters):
  # get data dimensions
  [N, D] = dat.shape
  # import ground truth
  original_class = ground_truth.original_class()

  # initialize using Kmeans
  centroids, labels = kmeans2(dat, clusters)
  idx, _ = vq(dat, centroids)

  # GMM
  # fit model
  start = timeit.default_timer()
  gmm = mixture.GaussianMixture(clusters)
  gmm.fit(dat)
  gmm_proba = gmm.predict_proba(dat)
  stop = timeit.default_timer()
  time = stop - start

  # get results
  ind = np.unravel_index(np.argmax(gmm_proba, axis=1), gmm_proba.shape)
  x = np.asarray(ind[1]).reshape(1, N)
  y_actu = pd.Series(np.asarray(original_class.tolist()).flatten(), name='Actual')
  y_pred = pd.Series(np.asarray(x.tolist()).flatten(), name='Predicted')
  df_confusion = pd.crosstab(y_actu, y_pred)
  print('GMM (%0.5f s)'%time)
  print(df_confusion, '\n')

  # VGMM
  # fit model
  start = timeit.default_timer()

  vgmm = mixture.BayesianGaussianMixture(clusters)
  vgmm.fit(dat)
  vgmm_proba = vgmm.predict_proba(dat)
  stop = timeit.default_timer()
  time = stop - start

  # get results
  ind2 = np.unravel_index(np.argmax(gmm_proba, axis=1), vgmm_proba.shape)
  x2 = np.asarray(ind2[1]).reshape(1, N)
  y_pred2 = pd.Series(np.asarray(x2.tolist()).flatten(), name='Predicted')
  df_confusion2 = pd.crosstab(y_actu, y_pred2)
  print('VGMM (%0.5f s)'%time)
  print(df_confusion2)


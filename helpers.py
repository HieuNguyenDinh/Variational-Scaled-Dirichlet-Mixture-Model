import numpy as np
from random import randint

# generate random ranges to find the most stable range for initialization
def v1_h1_gen():
  arr = np.array([1, 0.1, 0.1])

  for v1 in range(1, 1001):
    for h1 in range(1, 1001):

      test = np.array([randint(1, 1000) / 1000])
      test = np.append(test, randint(1, 1000) / 1000)
      test = np.append(test, randint(1, 1000) / 1000)

      arr = np.vstack([arr, test])
  return arr

# return unique values in list
def unique_in_list(a, D):
  res = []

  for i in a[0]:
    if i not in res:
      res.append(i)

  return len(res)

#check dimension of confusion matrix
def check_dimension(array):
  result = []
  for i in range(0, len(array)):
    count = 0
    if i == 0:
      result.append(array[i])
    else:
      for j in range(0, len(result)):
        if array[i] == result[j]:
          count = count + 1
      if count == 0:
        result.append(array[i])
  return len(result)

# Some common parameters
def common_paremeters():
  p1 = ([1, 1, 1])
  p2 = ([1, 1, 0.1])
  t = np.vstack((p1, p2))
  p3 = ([1, 1, 0.01])
  t = np.vstack((t, p3))
  p4 = ([1, 0.1, 1])
  t = np.vstack((t, p4))
  p5 = ([1, 0.01, 1])
  t = np.vstack((t, p5))
  p6 = ([1, 0.1, 0.1])
  t = np.vstack((t, p6))
  p7 = ([1, 0.01, 0.01])
  t = np.vstack((t, p7))
  p8 = ([1, 0.05, 0.05])
  t = np.vstack((t, p8))
  p9 = ([1, 0.1, 0.05])
  t = np.vstack((t, p9))
  p10 = ([1, 0.5, 0.01])
  t = np.vstack((t, p10))
  p11 = ([1, 0.05, 0.01])
  t = np.vstack((t, p11))
  p12 = ([1, 0.5, 0.11])
  t = np.vstack((t, p12))
  p13 = ([1, 0.555, .1])
  t = np.vstack((t, p13))
  p14 = ([1, 0.1, 0.01])
  t = np.vstack((t, p14))
  p15 = ([1, 0.15, 0.15])
  t = np.vstack((t, p15))
  p16 = ([1, 0.121, 0.011])
  t = np.vstack((t, p16))
  return t
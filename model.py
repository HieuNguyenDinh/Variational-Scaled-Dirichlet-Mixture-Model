import numpy as np
import scipy.special as ssp
import numpy.matlib as mat
import pandas as pd
from scipy.cluster.vq import kmeans,vq
from sklearn import preprocessing
import sys

import helpers, ground_truth


def SDMM(dat, u1, v1, h1, loop):
    # Avoid 0 since Scaled Dirichlet distribution points are positive
    dat = dat + sys.float_info.epsilon

    # get dimensions of dataset
    [N, D] = dat.shape

    # make data proportional
    dat_sum = dat.sum(axis=1).reshape([N, 1])
    dat = dat / dat_sum

    # Choose initial number of M components for K-Means
    M = 10
    centroids, labels = kmeans(dat, M)
    idx, _ = vq(dat, centroids)
    r_int = np.bincount(idx) / N

    # method of moments
    # a2 = np.ones([M, D])
    # # Method of moments
    # x_d = np.sum(dat)
    # x_2 = 0
    # for j in range(0, N):
    #     temp = dat[j, :]**2
    #     x_2 = x_2 + temp
    #
    # for j in range(0, M):
    #     num = (np.sum(dat, axis=0)/N - x_2/N) * x_d/N
    #     denom = x_2/N - (np.sum(dat, axis=0)**2) / N
    #     a2[j, :] = (num / denom)

    # Initialize pi & u & v & h(1 Dimension)
    pi = np.ones([1, M]) / M
    # values of u_int, v_int, h_int are random and chosen after a few tries
    u_int = np.ones([1, D]) * u1
    v_int = np.ones([1, D]) * v1
    h_int = np.ones([1, D]) * h1

    # a = u / v
    a_int = u_int / v_int

    # ln(a)
    ln_a = ssp.digamma(u_int) - np.log(v_int)

    # Sum of a
    sum_a = a_int.sum(axis=1)

    # b = Mean Dirichlet(h)
    b_int = h_int / h_int.sum(axis=1)

    # P(Cluster | x)
    Nk = N * r_int

    # Initialize [u & v & a] & [h & b]
    u = np.ones([M, D])
    v = np.ones([M, D])
    h = np.ones([M, D])

    # SumD(beta * x)
    b_x = mat.matmul(dat, np.transpose(b_int))

    # Initial value of u
    for l in range(0, D):
        # v[:, l] = v_int[0, l] - np.matmul(np.log(np.asarray(dat[:, l])), mat.repmat(r_int, N, 1))
        temp_u = ssp.digamma(sum_a) - ssp.digamma(a_int[0, l]) + \
                 mat.matmul(ssp.polygamma(1, sum_a), (
                     (mat.matmul(a_int, np.transpose(ln_a - np.log(a_int)))) - a_int[0, l] * (
                     ln_a[0, l] - np.log(a_int[0, l]))))
        u[:, l] = u_int[0, l] + np.transpose(Nk) * temp_u * a_int[0, l]

    # Repeat r_int N times
    r_int_N = mat.repmat(r_int, N, 1)

    # Initial value of v
    for j in range(0, M):
        temp_v = np.log(dat) + np.log(b_int[0, :]) - np.log(b_x)
        v[j, :] = v_int[0, :] - mat.matmul(np.transpose(r_int_N[:, j]), temp_v)

    a = u / v
    # a = a2

    # Intial value of h
    for j in range(0, M):
        temp_h = a_int[0, :] - a_int[0, :] * b_int[0, :] * dat / b_x
        h[j, :] = h_int[0, :] + mat.matmul(np.transpose(r_int_N[:, j]), temp_h)
    b = h / h.sum(axis=1).reshape([M, 1])
    # b = np.ones([M, D])

    # max value of loops for EM, which may varies for each model
    max = loop
    ctr = 0
    # EM loop
    for iter in range(0, max):
        eln_a = ssp.digamma(u) - np.log(v)
        eln_aa = ((ssp.digamma(u) - np.log(u)) ** 2) + ssp.polygamma(1, u)
        s_a = a.sum(axis=1)

        # Sum of G(a)
        s_galpha = ssp.loggamma(a).sum(axis=1)
        t3 = np.zeros([1, M])
        t4 = np.zeros([1, M])
        t5 = np.zeros([1, M])

        # SumD(a * ln(b))
        alogb = a * np.log(b)
        sum_alogb = alogb.sum(axis=1)

        # SumD((a-1) * ln(x))
        a_1_logx = mat.matmul(np.log(dat), np.transpose(a - 1))

        # SumD(beta * x)
        sum_beta_x = mat.matmul(dat, np.transpose(b))

        t1 = mat.repmat(np.log(pi), N, 1) + sum_alogb + a_1_logx - s_a * np.log(sum_beta_x)

        # ln(G(sum_a)) - Sum(G(a))
        t2 = ssp.loggamma(s_a) - s_galpha

        for l in range(0, D):
            t3 = t3 + (a[:, l] * (ssp.digamma(s_a) - ssp.digamma(a[:, l])) * (eln_a[:, l] - np.log(a[:, l])))
            t4 = t4 + (0.5 * (a[:, l] ** 2) * (ssp.polygamma(1, s_a) - ssp.polygamma(1, a[:, l])) * eln_aa[:, l])

        for j in range(0, M):
            t = 0
            for a1 in range(0, D):
                for b1 in range(0, D):
                    if (b1 != a1):
                        t += (a[j, a1] * a[j, b1] * ssp.polygamma(1, s_a[j]) * (eln_a[j, a1] - np.log(a[j, a1])) * \
                              (eln_a[j, b1] - np.log(a[j, b1])))
            t5[0, j] = t / 2

        R = t2 + t3 + t4 + t5
        lnRo = t1 + R

        # r
        ro = np.exp(lnRo)
        s_ro = ro.sum(axis=1)
        # avoid 0 and NAN values for next iterations
        log_r = np.log(ro + sys.float_info.epsilon) - np.log(np.reshape(s_ro, [N, 1]) + sys.float_info.epsilon)
        r = np.exp(log_r)
        # r = ro/np.reshape(s_ro, [N, 1])

        nk = r.sum(axis=0) + 1e-10
        r = r + 1e-10

        # update parameters a & b with new knowledge
        for j in range(0, M):
            temp_v = np.log(dat) + np.log(b[j, :]) - np.log(sum_beta_x[:, j].reshape([N, 1]))
            test = mat.matmul(np.transpose(r[:, j]), temp_v)
            v[j, :] = v_int[0, :] - mat.matmul(np.transpose(r[:, j]), temp_v)

        for l in range(0, D):
            temp = ssp.digamma(s_a) - ssp.digamma(a[:, l]) + \
                   (ssp.polygamma(1, s_a) * (
                       (a * (eln_a - np.log(a))).sum(axis=1) - (a[:, l] * (eln_a[:, l] - np.log(a[:, l])))))
            u[:, l] = mat.repmat(u_int[0, l], 1, M) + nk * temp * a[:, l]
        a = u / v

        for j in range(0, M):
            temp_h = a[j, :] - a[j, :] * b[j, :] * dat / sum_beta_x[:, j].reshape([N, 1])
            h[j, :] = h_int[0, :] + mat.matmul(np.transpose(r[:, j]), temp_h)

        sum_h = h.sum(axis=1)
        b = h / sum_h.reshape([M, 1])

        # update mixing weight pi
        pi = nk / N

        # print(ctr)
        # print(pi)
        ctr += 1

    # print accuracies
    original_class = ground_truth.original_class()
    ind = np.unravel_index(np.argmax(r, axis=1), r.shape)
    x = np.asarray(ind[1]).reshape(1, N)
    y_actu = pd.Series(np.asarray(original_class.tolist()).flatten(), name='Actual')
    y_pred = pd.Series(np.asarray(x.tolist()).flatten(), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)

    # Check dimensions of confusion matrix
    check = helpers.check_dimension(y_pred)
    if (check >= 2):
        return (df_confusion)
    else:
        return None

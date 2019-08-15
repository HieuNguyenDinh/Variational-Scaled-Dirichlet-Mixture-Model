import numpy as np
import timeit


import model, ground_truth, helpers, gmm

if __name__ == '__main__':
    # read data
    dat = np.genfromtxt('data/iris.csv', delimiter=',')

    # Test GMM & VGMM
    # g = ground_truth.original_class()
    # [n, d] = g.shape
    # cluster = helpers.unique_in_list(g, d)
    # gmm.GMM_VGMM(dat, cluster)
    # exit(0)

    # Choose parameters
    parameters = helpers.common_paremeters()
    # parameters = helpers.v1_h1_gen()

    # loop_test = ([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # write results to result.txt
    with open("result.txt", "w") as myfile:
        print('Running...')
        count = 0
        for i in range(0, len(parameters)):

            start = timeit.default_timer()
            # SDMM model(data, u, v, h, loop)
            a = model.SDMM(dat, parameters[i, 0], parameters[i, 1], parameters[i, 2], 100)
            stop = timeit.default_timer()
            time = stop - start


            # writes to file
            if a is not None:
                count = count + 1
                print(count)
                myfile.write("%s (%0.5f s)\n" %(parameters[i, :], time))
                myfile.write("%s\n" % a)
                myfile.write("\n")
                myfile.flush()

        print('DONE')






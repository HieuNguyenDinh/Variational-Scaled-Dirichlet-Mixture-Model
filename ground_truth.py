import numpy as np

# store ground truth for tested datasets
def original_class():
    # Iris
    original_class = np.concatenate((np.zeros([1, 50]), np.ones([1, 50]), np.ones([1, 50])*2), axis=1)

    # Vihicle
    # original_class = np.concatenate((np.ones([1, 218]), np.ones([1, 212]) * 2, np.ones([1, 217]) * 3, np.ones([1,199]) * 4), axis=1)

    # Breast Cancer
    # original_class = np.concatenate((np.ones([1, 458]), np.ones([1, 241]) * 2), axis=1)

    # Sonar
    # original_class = np.concatenate((np.ones([1, 97]), np.ones([1, 111]) * 2), axis=1)

    # Wine Quality
    # original_class = np.concatenate((np.zeros([1, 59]), np.ones([1, 71]), np.ones([1, 48]) * 2), axis=1)

    # Indian
    # original_class = np.concatenate((np.ones([1, 500]), np.ones([1, 268]) * 2), axis=1)

    return original_class

# SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Graphical Exploratory Data Analysis on Iris
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # == [ (Empirical) CUMULATION DISTRIBUTION FUNCTION START ] =================



import numpy as np



def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements
    """
    # (E)CDF = (Empirical) Cumulative Distribution Function
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y



# == [ (Empirical) CUMULATION DISTRIBUTION FUNCTION  END  ] ===================

# SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO-SHOWCASE-TWO
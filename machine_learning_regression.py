import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# for liearn regression, four assumptions: 
# 1. linear        if the graph is linear
# 2. normality          the residual plot, the dots are not clustered
# 3. independence       variables are independent to each other
# 4. homoskedastity     the residual are basically equal, no surge



# evaluate the linear regresion
# 1 mean absolute error (MAE)  mean of absolute residuals
# 2  mean squared error       mean of squarred residuals
# 3 root mean square error      root of mean square error
# 4 coefficients of determination       R2 = 1 -  RSS/TSS   sum of squared residuals  RSS, total sum of  squared mean residuals  TSS
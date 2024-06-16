import os
import tkinter as tk
from tkinter import ttk, filedialog

#
import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.stats import f
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
# from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer

# from sklearn.model_selection import train_test_split, cross_val_predict
# from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# from matplotlib.patches import Circle
class Classification:
    def __init__(self, df_x, df_y):

        self.df_x = df_x
        self.df_y = df_y

    def apply_classification(self, class_type, num_components, selected_label):
        ccircle = []
        eucl_dist = []

        if class_type == 'PCA':

            pca = PCA()
            # Fit PCA to standardized data
            pca.fit(StandardScaler().fit_transform(self.df_x))

            # Extract explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_

            pca = PCA(n_components=int(num_components) + 1)
            x_pca = pca.fit_transform(self.df_x)

            for j in range(self.df_x.values.shape[1]):
                corr1 = np.corrcoef(self.df_x.values[:, j], x_pca[:, 0])[0, 1]
                corr2 = np.corrcoef(self.df_x.values[:, j], x_pca[:, 1])[0, 1]
                ccircle.append((corr1, corr2))
                eucl_dist.append(np.sqrt(corr1 ** 2 + corr2 ** 2))

            return pca, x_pca, explained_variance_ratio, ccircle, eucl_dist

        elif class_type == 'LDA':
            # Discretize y into bins
            kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=200_000)
            y_binned = kbins.fit_transform(self.df_y[selected_label].values.reshape(-1, 1)).squeeze()

            lda = LinearDiscriminantAnalysis()
            lda.fit(self.df_x, y_binned)
            explained_variance_ratio = lda.explained_variance_ratio_

            lda = LinearDiscriminantAnalysis(n_components=int(num_components) + 1)
            x_lda = lda.fit_transform(self.df_x, y_binned)

            for j in range(self.df_x.values.shape[1]):
                corr1 = np.corrcoef(self.df_x.values[:, j], x_lda[:, 0])[0, 1]
                corr2 = np.corrcoef(self.df_x.values[:, j], x_lda[:, 1])[0, 1]
                ccircle.append((corr1, corr2))
                eucl_dist.append(np.sqrt(corr1 ** 2 + corr2 ** 2))

            return lda, x_lda, explained_variance_ratio, ccircle, eucl_dist, y_binned


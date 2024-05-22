import os
import tkinter as tk
from tkinter import ttk, filedialog

#
# import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.stats import f
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split, cross_val_predict
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
# from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# from matplotlib.patches import Circle
class PrincipalComponentAnalysis:
    def __init__(self, parent, df_X, df_y, class_type):
        self.y_binned = None
        self.x_lda = None
        self.lda = None
        self.selected_label = None
        self.class_type = class_type
        self.x_pc = 0
        self.y_pc = 1
        self.eucl_dist = []
        self.ccircle = []
        self.explained_variance_ratio = None
        self.x_pca = None
        self.pca = None
        self.parent = parent
        self.df_X = df_X
        self.df_y = df_y
        self.num_components = None

    def apply_classification(self):
        self.selected_label = self.select_label_combobox.get()
        self.x_pc = int(self.x_pc_entry.get()) - 1
        self.y_pc = int(self.y_pc_entry.get()) - 1

        self.num_components = max([self.x_pc, self.y_pc])
        if self.class_type == 'PCA':

            self.pca = PCA()
            # Fit PCA to standardized data
            self.pca.fit(StandardScaler().fit_transform(self.df_X))

            # Extract explained variance ratio
            self.explained_variance_ratio = self.pca.explained_variance_ratio_

            self.pca = PCA(n_components=int(self.num_components) + 1)
            self.x_pca = self.pca.fit_transform(self.df_X)

            for j in range(self.df_X.values.shape[1]):
                corr1 = np.corrcoef(self.df_X.values[:, j], self.x_pca[:, 0])[0, 1]
                corr2 = np.corrcoef(self.df_X.values[:, j], self.x_pca[:, 1])[0, 1]
                self.ccircle.append((corr1, corr2))
                self.eucl_dist.append(np.sqrt(corr1 ** 2 + corr2 ** 2))

            self.window.destroy()
class LinearDiscriminantAnalysis:
        elif self.class_type == 'LDA':
            # Discretize y into bins
            est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=200_000)
            self.y_binned = est.fit_transform(self.df_y[self.selected_label].values.reshape(-1, 1)).squeeze()

            self.lda = LinearDiscriminantAnalysis()
            self.x_lda = self.lda.fit(self.df_X, self.y_binned)
            self.explained_variance_ratio = self.lda.explained_variance_ratio_

            self.lda = LinearDiscriminantAnalysis(n_components=int(self.num_components) + 1)
            self.x_lda = self.lda.fit_transform(self.df_X, self.y_binned)

            for j in range(self.df_X.values.shape[1]):
                corr1 = np.corrcoef(self.df_X.values[:, j], self.x_lda[:, 0])[0, 1]
                corr2 = np.corrcoef(self.df_X.values[:, j], self.x_lda[:, 1])[0, 1]
                self.ccircle.append((corr1, corr2))
                self.eucl_dist.append(np.sqrt(corr1 ** 2 + corr2 ** 2))

            self.window.destroy()

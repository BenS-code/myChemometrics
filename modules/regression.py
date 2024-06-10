import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
from modules.preprocessing import DataStandardization


class PLS:

    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def train_test(self, test_ratio, groups):

        x_train, x_test, y_train, y_test = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

        if test_ratio != 0:
            splits = groups.nunique()

            # Define test split
            test_split = GroupShuffleSplit(n_splits=splits, test_size=test_ratio,
                                           train_size=1 - test_ratio, random_state=42)

            for train_index, test_index in test_split.split(self.df_x, self.df_y, groups):
                x_train, x_test = self.df_x.iloc[train_index], self.df_x.iloc[test_index]
                y_train, y_test = self.df_y.iloc[train_index], self.df_y.iloc[test_index]

        else:
            x_train, y_train = self.df_x.copy(), self.df_y.copy()

        return x_train, x_test, y_train, y_test

    def apply_pls(self, num_components, selected_labels, test_ratio, label_to_display, group_by, max_outliers, conf):

        groups = self.df_y[group_by]
        x_train, x_test, y_train, y_test = self.train_test(test_ratio, groups)

        num_train_groups = y_train[group_by].nunique()

        train_groups = y_train[group_by].copy()
        x_train = x_train.copy()
        y_train = y_train[selected_labels].copy()

        # Define cv data split
        cv_split = GroupKFold(n_splits=num_train_groups)

        parameters = {'n_components': np.arange(2, num_components + 1, 1)}

        pls = GridSearchCV(PLSRegression(scale=False), parameters, scoring='neg_mean_squared_error', verbose=1,
                           cv=cv_split)
        pls.fit(x_train, y_train, groups=train_groups)

        if max_outliers != 0:
            pls, x_train, y_train, train_groups, q_residuals, t_squared, q_residuals_conf, t_squared_conf = (
                self.remove_outliers(pls, x_train, y_train, train_groups, label_to_display,
                                     max_outliers, conf, parameters))
            cv_split = GroupKFold(n_splits=train_groups.nunique())

        else:
            pass

        y_pred_train = pls.predict(x_train)
        y_pred_train = pd.DataFrame(y_pred_train, columns=selected_labels)

        y_pred_cv = cross_val_predict(pls.best_estimator_, x_train, y_train, groups=train_groups,
                                      cv=cv_split)
        y_pred_cv = pd.DataFrame(y_pred_cv, columns=selected_labels)

        test_groups = pd.DataFrame([])
        if not x_test.empty:
            test_groups = y_test[group_by].copy()
            # Predict on test set
            y_pred_test = pls.predict(x_test)
            y_pred_test = pd.DataFrame(y_pred_test, columns=selected_labels)
        else:
            y_pred_test = pd.DataFrame([])

        return pls, y_train, y_test, y_pred_train, y_pred_cv, y_pred_test, train_groups, test_groups

    def remove_outliers(self, pls, x_train, y_train, train_groups, label_to_display, max_outliers, conf, parameters):

        ncomp = pls.best_estimator_.n_components

        # Get X scores
        scores = pls.best_estimator_.x_scores_

        # Get X loadings
        loadings = pls.best_estimator_.x_loadings_

        # Calculate error array
        err = x_train - np.dot(scores, loadings.T)

        # Calculate Q-residuals (sum over the rows of the error array)
        q_residuals = np.sum(err ** 2, axis=1)

        # Calculate Hotelling's T-squared (note that data are normalised by default)
        t_squared = np.sum((scores / np.std(scores, axis=0)) ** 2, axis=1)

        # Calculate confidence level for T-squared from the ppf of the F distribution
        t_squared_conf = (f.ppf(q=conf, dfn=ncomp, dfd=(x_train.shape[0] - ncomp))
                          * ncomp * (x_train.shape[0] - 1) / (x_train.shape[0] - ncomp))

        # Estimate the confidence level for the Q-residuals
        i = np.max(q_residuals) + 1
        while 1 - np.sum(q_residuals > i) / np.sum(q_residuals > 0) > conf:
            i -= 1
        q_residuals_conf = i

        # Sort the RMS distance from the origin in descending order (largest first)
        rms_dist = np.flip(np.argsort(np.sqrt(q_residuals ** 2 + t_squared ** 2)), axis=0)

        # Sort calibration spectra according to descending RMS distance
        Xc = x_train.iloc[rms_dist, :]
        Yc = y_train.iloc[rms_dist, :]
        pls_group_train = train_groups.iloc[rms_dist]

        # Discard one outlier at a time up to the value max_outliers
        # and calculate the rmse cross-validation of the PLS model

        # Define empty mse array
        rmse = np.zeros(max_outliers)

        for j in range(max_outliers):
            pls_temp = PLSRegression(n_components=ncomp, scale=True)
            pls_temp.fit(Xc.iloc[j:, :], Yc.iloc[j:, :])
            group_kfold = GroupKFold(n_splits=pls_group_train.iloc[j:].nunique() - 1)
            y_cv = cross_val_predict(pls_temp, Xc.iloc[j:, :], Yc.iloc[j:, :], groups=pls_group_train.iloc[j:],
                                     cv=group_kfold)
            y_cv = pd.DataFrame(y_cv, columns=Yc.columns)

            rmse[j] = np.sqrt(mean_squared_error(Yc[label_to_display].iloc[j:], y_cv[label_to_display]))

        # Find the position of the minimum in the mse (excluding the zeros)
        # rmsemin = np.where(rmse == np.min(rmse[np.nonzero(rmse)]))[0][0]
        rmsemin_index = np.argmin(rmse, axis=0)

        print(f'Removed {rmsemin_index} outliers')

        x_train = Xc.iloc[rmsemin_index:, :].copy()
        y_train = Yc.iloc[rmsemin_index:, :].copy()
        pls_group_train = pls_group_train.iloc[rmsemin_index:]

        cv_split = GroupKFold(n_splits=pls_group_train.nunique())

        pls = GridSearchCV(PLSRegression(scale=False), parameters,
                           scoring='neg_mean_squared_error', verbose=1, cv=cv_split)

        pls.fit(x_train, y_train, groups=pls_group_train)

        return pls, x_train, y_train, pls_group_train, q_residuals, t_squared, q_residuals_conf, t_squared_conf

    # def optimize_pcs(self, components_range, selected_labels, test_ratio, label_to_display):
    #
    #     rmse_cv_per_component = []
    #
    #     x_train, x_test, y_train, y_test = self.train_test(test_ratio, group_by)
    #
    #     # Iterate over different numbers of components
    #     for component in components_range:
    #         pls = PLSRegression(n_components=component, scale=False)
    #         # pls.fit(x_train, y_train)
    #
    #         # Define GroupKFold cross-validation
    #         group_kfold = GroupKFold(n_splits=y_train[label_to_display].nunique())
    #
    #         y_pred_cv = cross_val_predict(pls, x_train, y_train, groups=y_train[label_to_display], cv=group_kfold)
    #         rmse_cv = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    #
    #         rmse_cv_per_component.append(rmse_cv)
    #
    #     # Calculate and print the position of minimum in RMSE
    #     rmsemin = np.argmin(rmse_cv_per_component)
    #
    #     return rmsemin, rmse_cv_per_component

    # def optimize_pls(self, num_components, selected_labels, conf, max_outliers, test_ratio, label_to_display):
    #
    #     x_train, x_test, y_train, y_test = self.train_test(test_ratio, group_by)
    #
    #     x = pd.DataFrame(x_train, columns=self.df_x.columns)
    #     y = pd.DataFrame(y_train, columns=self.df_y.columns)
    #     y = y[selected_labels]
    #
    #     pls = PLSRegression(n_components=num_components, scale=True)
    #     pls.fit(x, y)
    #
    #     scores = pls.x_scores_
    #     loadings = pls.x_loadings_
    #
    #     error = x - np.dot(scores, loadings.T)
    #     q_residuals = np.sum(error ** 2, axis=1)
    #     t_squared = np.sum((scores / np.std(scores, axis=0)) ** 2, axis=1)
    #
    #     t_squared_conf = (f.ppf(q=conf, dfn=num_components,
    #                             dfd=(x.shape[0] - num_components))
    #                       * num_components * (x.shape[0] - 1) / (x.shape[0] - num_components))
    #
    #     q_residuals_conf = 0
    #
    #     i = np.max(q_residuals) + 1
    #     while 1 - np.sum(q_residuals > i) / np.sum(q_residuals > 0) > conf:
    #         i -= 1
    #         q_residuals_conf = i
    #
    #     rms_dist = np.flip(np.argsort(np.sqrt(q_residuals ** 2 + t_squared ** 2)), axis=0)
    #
    #     x_sorted = x.iloc[rms_dist, :]
    #     y_sorted = y.iloc[rms_dist, :]
    #
    #     self.df_x = x_sorted
    #     self.df_y = y_sorted
    #     x_train, x_test, y_train, y_test = self.train_test(test_ratio, group_by)
    #     x = pd.DataFrame(x_train, columns=self.df_x.columns)
    #     y = pd.DataFrame(y_train, columns=selected_labels)
    #
    #     rmse_cv = np.zeros(max_outliers)
    #
    #     for j in range(max_outliers):
    #         pls = PLSRegression(n_components=num_components, scale=False)
    #         pls.fit(x.iloc[j:, :], y.iloc[j:, :])
    #
    #         # Define GroupKFold cross-validation
    #         group_kfold = GroupKFold(n_splits=y[label_to_display].iloc[j:].nunique())
    #
    #         y_cv = cross_val_predict(pls, x.iloc[j:, :], y.iloc[j:, :],
    #                                  groups=y[label_to_display].iloc[j:], cv=group_kfold)
    #
    #         rmse = np.sqrt(mean_squared_error(y.iloc[j:, :], y_cv))
    #         rmse_cv[j] = rmse
    #
    #     rmsemin_index = np.argmin(rmse_cv, axis=0)
    #
    #     self.df_x = x.iloc[rmsemin_index:, :].copy()
    #     self.df_y = y.iloc[rmsemin_index:, :].copy()
    #
    #     return self.df_x, self.df_y, q_residuals, t_squared, q_residuals_conf, t_squared_conf, rmsemin_index

    def validate_pls(self, pls, selected_labels):

        coefficients = pls.coef_
        intercept = pls.intercept_
        y_predict = self.df_x @ coefficients.T + intercept
        y_predict = pd.DataFrame(np.array(y_predict), columns=[selected_labels])

        return y_predict
        #
        # self.apply_pls(num_components, )
        #
        # # Split data into train and test sets
        # self.x_train, self.x_test, self.y_train, self.y_test = \
        #     train_test_split(self.x_sorted.iloc[self.rmsemin_index:, :],
        #                      self.y_sorted.iloc[self.rmsemin_index:],
        #                      test_size=self.test_ratio)
        #
        # self.pls = PLSRegression(n_components=num_components)
        # # # Fit the model
        # self.pls.fit(self.x_train, self.y_train)
        #
        # self.y_pred_train = self.pls.predict(self.x_train)
        #
        # # Cross-validation
        # self.y_pred_cv = cross_val_predict(self.pls, self.x_train, self.y_train, cv=10)
        #
        # # Predict on test set
        # self.y_pred_test = self.pls.predict(self.x_test)
        #
        # self.rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        # self.rmse_cv = np.sqrt(mean_squared_error(self.y_train, self.y_pred_cv))
        # self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        #
        # # Calculate R^2 score
        # self.r2_train = r2_score(self.y_train, self.y_pred_train)
        # self.r2_cv = r2_score(self.y_train, self.y_pred_cv)
        # self.r2_test = r2_score(self.y_test, self.y_pred_test)
        #
        # self.window.destroy()

# class PLSOptimize:
#     def __init__(self, parent, df_x, df_y):
#         num_components = None
#         self.rmsemin_index = None
#         self.r2_score_cv = None
#         self.max_outliers = None
#         self.y_sorted = None
#         self.x_sorted = None
#         self.rms_dist = None
#         self.Q_conf = None
#         self.t_squared_conf = None
#         self.conf = None
#         self.t_squared = None
#         self.q_residuals= None
#         self.components_range = None
#         self.test_ratio = None
#         self.selected_label = None
#         self.top_left_plot = None
#         self.parent = parent
#         self.df_x = df_x
#         self.df_y = df_y
#         self.x_train = []
#         self.y_train = []
#         self.x_test = []
#         self.y_test = []
#         self.y_pred_train = None
#         self.y_pred_cv = None
#         self.y_pred_test = None
#         self.num_components = None
#         self.r2_train = None
#         self.r2_cv = None
#         self.r2_test = None
#         self.rmse_train = None
#         self.rmse_test = None
#         self.rmse_cv = None
#         self.pls = None
#         self.rmse_scores = []
#         self.r2_scores = []
#
#         self.window = tk.Toplevel(parent)
#         self.window.title("PLS Optimization Options")
#         # self.window.geometry("360x200")
#
#         # ComboBox for selecting label
#         self.select_label_label = ttk.Label(self.window, text="Select Label:")
#         self.select_label_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
#
#         self.select_label_combobox = ttk.Combobox(self.window, values=self.df_y.columns.tolist())
#         self.select_label_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
#         self.select_label_combobox.current(0)
#
#         self.test_ratio_label = ttk.Label(self.window, text="Test/Train ratio (0-0.5):")
#         self.test_ratio_label.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
#
#         self.test_ratio_entry = ttk.Entry(self.window)
#         self.test_ratio_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
#         self.test_ratio_entry.insert(tk.END, "0.1")
#
#         # Label and entry for number of components
#         self.num_components_label = ttk.Label(self.window, text="Number of Components:")
#         self.num_components_label.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
#
#         self.num_components_entry = ttk.Entry(self.window)
#         self.num_components_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
#         self.num_components_entry.insert(tk.END, "2")
#
#         # Label and entry for number of components
#         self.num_outliers_label = ttk.Label(self.window, text="Number of Outliers to Remove:")
#         self.num_outliers_label.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
#
#         self.num_outliers_entry = ttk.Entry(self.window)
#         self.num_outliers_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
#         self.num_outliers_entry.insert(tk.END, "20")
#
#         # Label and entry for number of components
#         self.confidence_label = ttk.Label(self.window, text="Confidence Level (0.5-0.99):")
#         self.confidence_label.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
#
#         self.confidence_entry = ttk.Entry(self.window)
#         self.confidence_entry.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
#         self.confidence_entry.insert(tk.END, "0.95")
#
#         # optimize with outliers removal button
#         self.optimize_button = ttk.Button(self.window, text="Optimize", command=self.optimize_pls)
#         self.optimize_button.grid(row=5, column=0, padx=5, pady=5)
#
#         # Apply and Cancel buttons
#         self.opt_pc_button = ttk.Button(self.window, text="Optimize PCs", command=self.optimize_pls_comp)
#         self.opt_pc_button.grid(row=2, column=2, padx=5, pady=5)
#
#         self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
#         self.cancel_button.grid(row=5, column=1, padx=5, pady=5)
#
#     def optimize_pls(self):
#         self.selected_label = self.select_label_combobox.get()
#         self.test_ratio = float(self.test_ratio_entry.get())
#         num_components = int(self.num_components_entry.get())
#         pls = PLSRegression(n_components=num_components)
#         pls.fit(self.df_x, self.df_y)
#
#         scores= pls.x_scores_
#         loadings= pls.x_loadings_
#
#         error = self.df_x - np.dot(scores P.T)
#         self.q_residuals= np.sum(error ** 2, axis=1)
#         self.t_squared = np.sum((pls.x_scores_ / np.std(pls.x_scores_, axis=0)) ** 2, axis=1)
#
#         self.conf = float(self.confidence_entry.get())
#         self.t_squared_conf = (f.ppf(q=self.conf, dfn=num_components,
#                                dfd=(self.df_x.shape[0] - num_components))
#                          * num_components * (self.df_x.shape[0] - 1) / (self.df_x.shape[0] - num_components))
#
#         i = np.max(self.Q) + 1
#         while 1 - np.sum(self.q_residuals> i) / np.sum(self.q_residuals> 0) > self.conf:
#             i -= 1
#             self.Q_conf = i
#
#         self.rms_dist = np.flip(np.argsort(np.sqrt(self.q_residuals** 2 + self.t_squared ** 2)), axis=0)
#
#         self.x_sorted = self.df_x.iloc[self.rms_dist, :]
#         self.y_sorted = self.df_y.iloc[self.rms_dist, :][self.selected_label].copy()
#
#         self.max_outliers = int(self.num_outliers_entry.get())
#
#         self.rmse_cv = np.zeros(self.max_outliers)
#         self.r2_score_cv = np.zeros(self.max_outliers)
#
#         for j in range(self.max_outliers):
#             pls = PLSRegression(n_components=num_components)
#             pls.fit(self.x_sorted.iloc[j:, :], self.y_sorted.iloc[j:])
#             y_cv = cross_val_predict(pls, self.x_sorted.iloc[j:, :], self.y_sorted.iloc[j:], cv=10)
#
#             self.rmse_cv[j] = np.sqrt(mean_squared_error(self.y_sorted.iloc[j:], y_cv))
#             self.r2_score_cv[j] = r2_score(self.y_sorted.iloc[j:], y_cv)
#
#         self.rmsemin_index = np.argmin(self.rmse_cv, axis=0)
#
#         self.df_x = self.df_x.iloc[self.rmsemin_index:, :].copy()
#         self.df_y = self.df_y.iloc[self.rmsemin_index:, :].copy()
#
#         # Split data into train and test sets
#         self.x_train, self.x_test, self.y_train, self.y_test = \
#             train_test_split(self.x_sorted.iloc[self.rmsemin_index:, :],
#                              self.y_sorted.iloc[self.rmsemin_index:],
#                              test_size=self.test_ratio)
#
#         self.pls = PLSRegression(n_components=num_components)
#         # # Fit the model
#         self.pls.fit(self.x_train, self.y_train)
#
#         self.y_pred_train = self.pls.predict(self.x_train)
#
#         # Cross-validation
#         self.y_pred_cv = cross_val_predict(self.pls, self.x_train, self.y_train, cv=10)
#
#         # Predict on test set
#         self.y_pred_test = self.pls.predict(self.x_test)
#
#         self.rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
#         self.rmse_cv = np.sqrt(mean_squared_error(self.y_train, self.y_pred_cv))
#         self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
#
#         # Calculate R^2 score
#         self.r2_train = r2_score(self.y_train, self.y_pred_train)
#         self.r2_cv = r2_score(self.y_train, self.y_pred_cv)
#         self.r2_test = r2_score(self.y_test, self.y_pred_test)
#
#         self.window.destroy()
#
#     def optimize_pls_comp(self):
#
#         self.num_components = int(self.num_components_entry.get())
#         self.selected_label = self.select_label_combobox.get()
#         self.test_ratio = float(self.test_ratio_entry.get())
#         if self.test_ratio < 0.001:
#             self.test_ratio = 0.001
#         if self.test_ratio > 0.5:
#             self.test_ratio = 0.5
#
#         self.selected_label = self.select_label_combobox.get()
#
#         if self.num_components > self.df_x.shape[1]:
#             self.num_components = self.df_x.shape[1]
#
#         self.components_range = np.arange(1, 20)
#         if max(self.components_range) > self.df_x.shape[1]:
#             self.components_range = np.arange(1, self.df_x.shape[1])
#
#         self.rmse_scores = []
#
#         # Iterate over different numbers of components
#         for component in self.components_range:
#             # Train the PLS model
#             pls = PLSRegression(n_components=component)
#             # pls.fit(self.x_train, self.y_train)
#
#             # Cross-validation
#             y_cv = cross_val_predict(pls, self.df_x, self.df_y[self.selected_label], cv=10)
#
#             # Calculate RMSE and append to the list
#             rmse = np.sqrt(mean_squared_error(self.df_y[self.selected_label], y_cv))
#             self.rmse_scores.append(rmse)
#
#             # Calculate R2 score and append to the list
#             r2 = r2_score(self.df_y[self.selected_label], y_cv)
#             self.r2_scores.append(r2)
#
#         # Calculate and print the position of minimum in RMSE
#         rmsemin = np.argmin(self.rmse_scores)
#
#         self.num_components_entry.delete(0, tk.END)
#         self.num_components_entry.insert(tk.END, rmsemin + 1)
#
#
#

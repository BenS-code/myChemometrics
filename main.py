import _tkinter
import os
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle


class SelectColumnsWindow:
    def __init__(self, parent, df_raw):
        self.parent = parent
        self.columns = df_raw.columns
        self.selected_columns = []

        self.window = tk.Toplevel(parent)
        self.window.title("Select Columns")
        self.window.geometry("400x400")

        self.listbox = tk.Listbox(self.window, selectmode=tk.EXTENDED)
        for col in self.columns:
            self.listbox.insert(tk.END, col)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.listbox, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)

        # self.listbox.bind("<<ListboxSelect>>", self.on_select)

        self.select_button = ttk.Button(self.window, text="Select", command=self.select_columns)
        self.select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.select_all_button = ttk.Button(self.window, text="Mark All", command=self.select_all)
        self.select_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def select_all(self):
        self.listbox.select_set(0, tk.END)

    def select_columns(self):
        selected_indices = self.listbox.curselection()
        self.selected_columns = [self.columns[idx] for idx in selected_indices]

        self.window.destroy()


class FilterData:
    def __init__(self, parent, df_X, df_y):
        self.x_threshold = 3.0
        self.y_threshold = 3.0
        self.df_X = df_X
        self.df_y = df_y
        self.selected_x_columns = self.df_X.columns
        self.selected_y_columns = self.df_y.columns
        self.df_temp = pd.concat([self.df_y, self.df_X], axis=1)
        self.parent = parent

        self.window = tk.Toplevel(parent)
        self.window.title("Filter Data")
        # self.window.geometry("500x200")

        ttk.Label(self.window,
                  text="Remove above \nthreshold (# STD):").grid(row=1, column=0, padx=5, pady=10, sticky='ew')

        self.x_std_num_label = ttk.Label(self.window, text="X Correlation Distance")
        self.x_std_num_label.grid(row=0, column=1, padx=5, pady=10, sticky='ew')

        self.x_std_num_entry = ttk.Entry(self.window)
        self.x_std_num_entry.grid(row=1, column=1, padx=5, pady=10, sticky='ew')
        self.x_std_num_entry.insert(tk.END, "3")

        # Label and entry for number of components
        self.y_std_num_label = ttk.Label(self.window, text="y Outliers")
        self.y_std_num_label.grid(row=0, column=2, padx=5, pady=10, sticky='ew')

        self.y_std_num_entry = ttk.Entry(self.window)
        self.y_std_num_entry.grid(row=1, column=2, padx=5, pady=10, sticky='ew')
        self.y_std_num_entry.insert(tk.END, "3")

        self.filter_button = ttk.Button(self.window, text="Filter", command=self.filter_raw_data)
        self.filter_button.grid(row=3, column=0, padx=5, pady=10, sticky='ew')

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=3, column=1, padx=5, pady=10, sticky='ew')

    def filter_raw_data(self):
        self.x_threshold = float(self.x_std_num_entry.get())
        self.y_threshold = float(self.y_std_num_entry.get())

        dissimilarities = cdist(self.df_temp[self.selected_x_columns], self.df_temp[self.selected_x_columns],
                                metric="correlation")

        normalized_dissimilarities_vector = (dissimilarities[1] - np.mean(dissimilarities[1])) / np.std(
            dissimilarities[1])

        rows_to_remove = np.where(np.abs(normalized_dissimilarities_vector) > self.x_threshold)

        self.df_temp = self.df_temp.drop(index=rows_to_remove[0].tolist())

        for col in self.df_temp[self.selected_y_columns].columns:
            mean = self.df_temp[col].mean()
            std = self.df_temp[col].std()
            self.df_temp = self.df_temp[(self.df_temp[col] - mean).abs() <= (self.y_threshold * std)]

        self.df_temp = self.df_temp.dropna()
        self.df_temp = self.df_temp.reset_index()

        self.window.destroy()


class PLS:
    def __init__(self, parent, df_X, df_y):

        self.rmse_test_opt = None
        self.rmse_cv_opt = None
        self.rmse_train_opt = None
        self.r2_test_opt = None
        self.r2_cv_opt = None
        self.r2_train_opt = None
        self.y_pred_test_opt = None
        self.y_pred_cv_opt = None
        self.y_pred_train_opt = None
        self.pls_opt = None
        self.train_test_ratio = None
        self.selected_label = None
        self.top_left_plot = None
        self.parent = parent
        self.df_X = df_X
        self.df_y = df_y
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_pred_train = None
        self.y_pred_cv = None
        self.y_pred_test = None
        self.num_components = None
        self.r2_train = None
        self.r2_cv = None
        self.r2_test = None
        self.rmse_train = None
        self.rmse_test = None
        self.rmse_cv = None
        self.pls = None
        self.rmse_scores = []
        self.r2_scores = []

        self.window = tk.Toplevel(parent)
        self.window.title("PLS Options")
        # self.window.geometry("360x200")

        # Label and entry for number of components
        self.num_components_label = ttk.Label(self.window, text="Number of Components:")
        self.num_components_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.num_components_entry = ttk.Entry(self.window)
        self.num_components_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.num_components_entry.insert(tk.END, "2")

        # ComboBox for selecting label
        self.select_label_label = ttk.Label(self.window, text="Select Label:")
        self.select_label_label.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.select_label_combobox = ttk.Combobox(self.window, values=self.df_y.columns.tolist())
        self.select_label_combobox.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.select_label_combobox.current(0)

        self.train_test_ratio_label = ttk.Label(self.window, text="Test/Train ratio (0-0.5):")
        self.train_test_ratio_label.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        self.train_test_ratio_entry = ttk.Entry(self.window)
        self.train_test_ratio_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        self.train_test_ratio_entry.insert(tk.END, "0.1")

        # Apply and Cancel buttons
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_pls)
        self.apply_button.grid(row=3, column=0, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=3, column=1, padx=5, pady=5)

    def apply_pls(self):
        self.num_components = int(self.num_components_entry.get())
        self.selected_label = self.select_label_combobox.get()
        self.train_test_ratio = float(self.train_test_ratio_entry.get())
        if self.train_test_ratio < 0.001:
            self.train_test_ratio = 0.001
        if self.train_test_ratio > 0.5:
            self.train_test_ratio = 0.5

        # Split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.df_X, self.df_y[self.select_label_combobox.get()],
                             test_size=self.train_test_ratio)

        if self.num_components > self.df_X.shape[1]:
            self.num_components = self.df_X.shape[1]

        # Initialize PLS model with desired number of components
        self.pls = PLSRegression(n_components=self.num_components)

        # Fit the model
        self.pls.fit(self.x_train, self.y_train)

        self.y_pred_train = self.pls.predict(self.x_train)

        # Cross-validation
        self.y_pred_cv = cross_val_predict(self.pls, self.x_train, self.y_train, cv=10)

        # Predict on test set
        self.y_pred_test = self.pls.predict(self.x_test)

        # Calculate R^2 score
        self.r2_train = r2_score(self.y_train, self.y_pred_train)
        self.r2_cv = r2_score(self.y_train, self.y_pred_cv)
        self.r2_test = r2_score(self.y_test, self.y_pred_test)

        self.rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        self.rmse_cv = np.sqrt(mean_squared_error(self.y_train, self.y_pred_cv))
        self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))

        self.window.destroy()


class PLSOptimize:
    def __init__(self, parent, df_X, df_y):
        self.rmse_test_opt = None
        self.rmse_cv_opt = None
        self.rmse_train_opt = None
        self.r2_test_opt = None
        self.r2_cv_opt = None
        self.r2_train_opt = None
        self.y_pred_test_opt = None
        self.y_pred_cv_opt = None
        self.y_pred_train_opt = None
        self.pls_opt = None
        self.components_range = None
        self.train_test_ratio = None
        self.selected_label = None
        self.top_left_plot = None
        self.parent = parent
        self.df_X = df_X
        self.df_y = df_y
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_pred_train = None
        self.y_pred_cv = None
        self.y_pred_test = None
        self.num_components = None
        self.r2_train = None
        self.r2_cv = None
        self.r2_test = None
        self.rmse_train = None
        self.rmse_test = None
        self.rmse_cv = None
        self.pls = None
        self.rmse_scores = []
        self.r2_scores = []

        self.window = tk.Toplevel(parent)
        self.window.title("PLS Optimization Options")
        # self.window.geometry("360x200")

        # ComboBox for selecting label
        self.select_label_label = ttk.Label(self.window, text="Select Label:")
        self.select_label_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.select_label_combobox = ttk.Combobox(self.window, values=self.df_y.columns.tolist())
        self.select_label_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.select_label_combobox.current(0)

        self.train_test_ratio_label = ttk.Label(self.window, text="Test/Train ratio (0-0.5):")
        self.train_test_ratio_label.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.train_test_ratio_entry = ttk.Entry(self.window)
        self.train_test_ratio_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.train_test_ratio_entry.insert(tk.END, "0.1")

        # Label and entry for number of components
        self.num_components_label = ttk.Label(self.window, text="Number of Components:")
        self.num_components_label.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        self.num_components_entry = ttk.Entry(self.window)
        self.num_components_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        self.num_components_entry.insert(tk.END, "2")

        # Label and entry for number of components
        self.num_outliers_label = ttk.Label(self.window, text="Number of Outliers to Remove:")
        self.num_outliers_label.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

        self.num_outliers_entry = ttk.Entry(self.window)
        self.num_outliers_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        self.num_outliers_entry.insert(tk.END, "20")

        # Label and entry for number of components
        self.confidence_label = ttk.Label(self.window, text="Confidence Level (0.5-0.99):")
        self.confidence_label.grid(row=4, column=0, padx=5, pady=5, sticky='ew')

        self.confidence_entry = ttk.Entry(self.window)
        self.confidence_entry.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
        self.confidence_entry.insert(tk.END, "0.95")

        # outliers removal button
        self.outliers_button = ttk.Button(self.window, text="Remove Outliers", command=self.outlier_removal)
        self.outliers_button.grid(row=5, column=0, padx=5, pady=5)

        # Apply and Cancel buttons
        self.opt_pc_button = ttk.Button(self.window, text="Optimize PCs", command=self.optimize_pls_comp)
        self.opt_pc_button.grid(row=2, column=2, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=5, column=1, padx=5, pady=5)

    def outlier_removal(self):
        pass

    def optimize_pls_comp(self):

        self.num_components = int(self.num_components_entry.get())
        self.selected_label = self.select_label_combobox.get()
        self.train_test_ratio = float(self.train_test_ratio_entry.get())
        if self.train_test_ratio < 0.001:
            self.train_test_ratio = 0.001
        if self.train_test_ratio > 0.5:
            self.train_test_ratio = 0.5

        # Split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.df_X, self.df_y[self.select_label_combobox.get()],
                             test_size=self.train_test_ratio)

        if self.num_components > self.df_X.shape[1]:
            self.num_components = self.df_X.shape[1]

        self.components_range = range(1, 20)
        if max(self.components_range) > self.df_X.shape[1]:
            self.components_range = range(1, self.df_X.shape[1])

        # Iterate over different numbers of components
        for n_components in self.components_range:
            # Train the PLS model
            pls = PLSRegression(n_components=n_components)
            pls.fit(self.x_train, self.y_train)

            # Cross-validation
            y_cv = cross_val_predict(pls, self.x_train, self.y_train, cv=10)

            # Calculate RMSE and append to the list
            rmse = np.sqrt(mean_squared_error(self.y_train, y_cv))
            self.rmse_scores.append(rmse)

            # Calculate R2 score and append to the list
            r2 = r2_score(self.y_train, y_cv)
            self.r2_scores.append(r2)

        # Calculate and print the position of minimum in RMSE
        rmsemin = np.argmin(self.rmse_scores)

        self.num_components_entry.delete(0, tk.END)
        self.num_components_entry.insert(tk.END, rmsemin + 1)

        # # Define PLS object with optimal number of components
        # self.pls_opt = PLSRegression(n_components=rmsemin + 1)
        #
        # # Fir to the entire dataset
        # self.pls_opt.fit(self.x_train, self.y_train)
        #
        # self.y_pred_train_opt = self.pls_opt.predict(self.x_train)
        #
        # # Cross-validation
        # self.y_pred_cv_opt = cross_val_predict(self.pls_opt, self.x_train, self.y_train, cv=10)
        #
        # self.y_pred_test_opt = self.pls_opt.predict(self.x_test)
        #
        # # Calculate scores for calibration and cross-validation
        # self.r2_train_opt = r2_score(self.y_train, self.y_pred_train_opt)
        # self.r2_cv_opt = r2_score(self.y_train, self.y_pred_cv_opt)
        # self.r2_test_opt = r2_score(self.y_train, self.y_pred_test_opt)
        #
        # # Calculate mean squared error for calibration and cross validation
        # self.rmse_train_opt = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train_opt))
        # self.rmse_cv_opt = np.sqrt(mean_squared_error(self.y_train, self.y_pred_cv_opt))
        # self.rmse_test_opt = np.sqrt(mean_squared_error(self.y_train, self.y_pred_test_opt))

        # # Plot regression and figures of merit
        # rangey = max(y) - min(y)
        # rangex = max(y_c) - min(y_c)


class Classification:
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

        self.window = tk.Toplevel(parent)
        self.window.title("Classification Options")
        # self.window.geometry("360x200")

        # Label and entry for number of components
        self.x_pc_label = ttk.Label(self.window, text="x axis component:")
        self.x_pc_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.x_pc_entry = ttk.Entry(self.window)
        self.x_pc_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.x_pc_entry.insert(tk.END, "1")

        # Label and entry for number of components
        self.y_pc_label = ttk.Label(self.window, text="y axis component:")
        self.y_pc_label.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.y_pc_entry = ttk.Entry(self.window)
        self.y_pc_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.y_pc_entry.insert(tk.END, "2")

        # ComboBox for selecting label
        self.select_label_label = ttk.Label(self.window, text="Select Label:")
        self.select_label_label.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        self.select_label_combobox = ttk.Combobox(self.window, values=self.df_y.columns.tolist())
        self.select_label_combobox.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        self.select_label_combobox.current(0)

        # Apply and Cancel buttons
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_classification)
        self.apply_button.grid(row=3, column=0, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=3, column=1, padx=5, pady=5)

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


class MyChemometrix:
    def __init__(self, master):
        super().__init__()

        self.selected_x_columns = []
        self.selected_y_columns = []
        self.master = master
        self.master.title("MyChemometrix")

        # Get screen size
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        self.master.geometry(f'{screen_width}x{screen_height}')

        # Calculate frame sizes
        buttons_frame_height = screen_height // 25
        status_frame_height = screen_height // 25
        data_frame_width = screen_width * 0.25
        graphs_frame_width = screen_width * 0.75

        # Styles
        self.style = ttk.Style()
        self.style.theme_use("Breeze")

        # self.style.configure("TFrame", background="#F0F0F0")
        # self.style.configure("TLabelFrame", background="#F0F0F0")
        self.style.configure("TButton", background="light blue", foreground="black", padding=5)
        self.style.map("TButton", background=[("active", "light sky blue")])

        # Frames
        self.buttons_frame = ttk.Frame(master, height=buttons_frame_height)
        self.buttons_frame.pack(side="top", fill="x")

        self.status_frame = ttk.Frame(master, height=status_frame_height)
        self.status_frame.pack(side="bottom", fill="x")

        # PanedWindow for resizable frames
        self.paned_window = ttk.PanedWindow(master, orient="horizontal")
        self.paned_window.pack(expand=True, fill="both")

        self.data_frame = ttk.LabelFrame(self.paned_window, text="Data", width=data_frame_width)
        self.paned_window.add(self.data_frame)

        self.graphs_frame = ttk.LabelFrame(self.paned_window, text="Graphics",
                                           width=graphs_frame_width)
        self.paned_window.add(self.graphs_frame)

        self.fig1 = plt.Figure(figsize=(5, 5), dpi=100)
        self.fig2 = plt.Figure(figsize=(5, 5), dpi=100)
        self.fig3 = plt.Figure(figsize=(5, 5), dpi=100)
        self.fig4 = plt.Figure(figsize=(5, 5), dpi=100)
        # Canvas areas in the graphs_frame
        self.top_left_canvas = tk.Canvas(self.graphs_frame,
                                         width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_left_canvas.grid(row=0, column=0, sticky="nsew")

        self.top_left_plot = FigureCanvasTkAgg(self.fig1, master=self.top_left_canvas)
        self.top_left_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        # Add line frame around the canvas widget
        self.top_left_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.top_left_plot.draw()

        self.top_right_canvas = tk.Canvas(self.graphs_frame,
                                          width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_right_canvas.grid(row=0, column=1, sticky="nsew")

        self.top_right_plot = FigureCanvasTkAgg(self.fig2, master=self.top_right_canvas)
        self.top_right_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.top_right_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.top_right_plot.draw()

        self.bottom_left_canvas = tk.Canvas(self.graphs_frame,
                                            width=graphs_frame_width // 2, height=screen_height // 2)
        self.bottom_left_canvas.grid(row=1, column=0, sticky="nsew")

        self.bottom_left_plot = FigureCanvasTkAgg(self.fig3, master=self.bottom_left_canvas)
        self.bottom_left_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.bottom_left_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.bottom_left_plot.draw()

        self.bottom_right_canvas = tk.Canvas(self.graphs_frame,
                                             width=graphs_frame_width // 2, height=screen_height // 2)
        self.bottom_right_canvas.grid(row=1, column=1, sticky="nsew")

        self.bottom_right_plot = FigureCanvasTkAgg(self.fig4, master=self.bottom_right_canvas)
        self.bottom_right_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.bottom_right_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.bottom_right_plot.draw()

        # Configure row and column weights for resizing
        self.graphs_frame.grid_rowconfigure(0, weight=1)
        self.graphs_frame.grid_rowconfigure(1, weight=1)
        self.graphs_frame.grid_columnconfigure(0, weight=1)
        self.graphs_frame.grid_columnconfigure(1, weight=1)

        # Create and place Treeview widgets inside LabelFrames
        self.raw_data_frame = ttk.LabelFrame(self.data_frame, text="Raw Data")
        self.raw_data_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.raw_data_tree = ttk.Treeview(self.raw_data_frame, show='headings')
        self.raw_data_tree.pack(side="left", fill="both", expand=True)

        self.raw_data_scroll_y = tk.Scrollbar(self.raw_data_tree, orient="vertical",
                                              command=self.raw_data_tree.yview)
        self.raw_data_scroll_y.pack(side="right", fill="y")
        self.raw_data_tree.config(yscrollcommand=self.raw_data_scroll_y.set)

        self.raw_data_scroll_x = tk.Scrollbar(self.raw_data_tree, orient="horizontal",
                                              command=self.raw_data_tree.xview)
        self.raw_data_scroll_x.pack(side="bottom", fill="x")
        self.raw_data_tree.config(xscrollcommand=self.raw_data_scroll_x.set)

        self.labels_data_frame = ttk.LabelFrame(self.data_frame, text="Labels")
        self.labels_data_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.labels_data_tree = ttk.Treeview(self.labels_data_frame, show='headings')
        self.labels_data_tree.pack(side="left", fill="both", expand=True)

        self.labels_data_scroll_y = tk.Scrollbar(self.labels_data_tree, orient="vertical",
                                                 command=self.labels_data_tree.yview)
        self.labels_data_scroll_y.pack(side="right", fill="y")

        self.labels_data_tree.config(yscrollcommand=self.labels_data_scroll_y.set)

        self.labels_data_scroll_x = tk.Scrollbar(self.labels_data_tree, orient="horizontal",
                                                 command=self.labels_data_tree.xview)
        self.labels_data_scroll_x.pack(side="bottom", fill="x")

        self.labels_data_tree.config(xscrollcommand=self.labels_data_scroll_x.set)

        self.features_data_frame = ttk.LabelFrame(self.data_frame, text="Features")
        self.features_data_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.features_data_tree = ttk.Treeview(self.features_data_frame, show='headings')
        self.features_data_tree.pack(side="left", fill="both", expand=True)

        self.features_data_scroll_y = tk.Scrollbar(self.features_data_tree, orient="vertical",
                                                   command=self.features_data_tree.yview)
        self.features_data_scroll_y.pack(side="right", fill="y")

        self.features_data_tree.config(yscrollcommand=self.features_data_scroll_y.set)

        self.features_data_scroll_x = tk.Scrollbar(self.features_data_tree, orient="horizontal",
                                                   command=self.features_data_tree.xview)
        self.features_data_scroll_x.pack(side="bottom", fill="x")

        self.features_data_tree.config(xscrollcommand=self.features_data_scroll_x.set)

        # Divide buttons_frame into four sub-frames
        self.data_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Data")
        self.data_buttons_frame.pack(side="left", fill="both", expand=True)

        self.preprocessing_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Preprocessing")
        self.preprocessing_buttons_frame.pack(side="left", fill="both", expand=True)

        self.regression_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Regression")
        self.regression_buttons_frame.pack(side="left", fill="both", expand=True)

        self.classifier_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Classification")
        self.classifier_buttons_frame.pack(side="left", fill="both", expand=True)

        # Create buttons
        self.load_data_button = ttk.Button(self.data_buttons_frame, text="Load Data",
                                           command=self.import_file)
        self.load_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.select_y_button = ttk.Button(self.data_buttons_frame, text="Select y", state="disabled",
                                          command=self.open_select_labels_window)
        self.select_y_button.pack(side="left", fill="both", padx=5, pady=5)

        self.select_x_button = ttk.Button(self.data_buttons_frame, text="Select X", state="disabled",
                                          command=self.open_select_features_window)
        self.select_x_button.pack(side="left", fill="both", padx=5, pady=5)

        self.display_data_button = ttk.Button(self.data_buttons_frame, text="Display Data", state="disabled",
                                              command=self.display_xy)
        self.display_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.filter_data_button = ttk.Button(self.preprocessing_buttons_frame, text="Filter Data", state="disabled",
                                             command=self.filter_data)
        self.filter_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.standardscaler_button = ttk.Button(self.preprocessing_buttons_frame, text="Standardize Variables",
                                                state="disabled",
                                                command=self.apply_labels_normalization)
        self.standardscaler_button.pack(side="left", fill="both", padx=5, pady=5)

        self.msc_button = ttk.Button(self.preprocessing_buttons_frame, text="MSC", state="disabled",
                                     command=self.apply_msc)
        self.msc_button.pack(side="left", fill="both", padx=5, pady=5)

        self.snv_button = ttk.Button(self.preprocessing_buttons_frame, text="SNV", state="disabled",
                                     command=self.apply_snv)
        self.snv_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pls_button = ttk.Button(self.regression_buttons_frame, text="PLS", state="disabled",
                                     command=self.open_pls_window)
        self.pls_button.pack(side="left", fill="both", padx=5, pady=5)

        self.optimize_button = ttk.Button(self.regression_buttons_frame, text="Optimize", state="disabled",
                                          command=self.optimize_pls_window)
        self.optimize_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pca_button = ttk.Button(self.classifier_buttons_frame, text="PCA", state="disabled",
                                     command=lambda: self.open_classification_window('PCA'))
        self.pca_button.pack(side="left", fill="both", padx=5, pady=5)

        self.LDA_button = ttk.Button(self.classifier_buttons_frame, text="LDA",
                                     command=lambda: self.open_classification_window('LDA'),
                                     state="disabled")
        self.LDA_button.pack(side="left", fill="both", padx=5, pady=5)

        self.df_raw = None
        self.df_X = None
        self.df_y = None

        self.x_rows = tk.StringVar()
        self.x_cols = tk.StringVar()
        self.y_rows = tk.StringVar()
        self.y_cols = tk.StringVar()
        self.x_rows.set("")
        self.x_cols.set("")
        self.y_rows.set("")
        self.y_cols.set("")

        self.x_rows_stat_label = tk.Label(self.status_frame, text="X rows")
        self.x_rows_stat_label.grid(row=0, column=0, sticky="nsew")

        self.x_rows_stat_entry = tk.Entry(self.status_frame, textvariable=self.x_rows, state='readonly')
        self.x_rows_stat_entry.grid(row=0, column=1, sticky="nsew")

        self.x_cols_stat_label = tk.Label(self.status_frame, text="X columns")
        self.x_cols_stat_label.grid(row=0, column=2, sticky="nsew")

        self.x_cols_stat_entry = tk.Entry(self.status_frame, textvariable=self.x_cols, state='readonly')
        self.x_cols_stat_entry.grid(row=0, column=3, sticky="nsew")

        self.y_rows_stat_label = tk.Label(self.status_frame, text="y rows")
        self.y_rows_stat_label.grid(row=0, column=4, sticky="nsew")

        self.y_rows_stat_entry = tk.Entry(self.status_frame, textvariable=self.y_rows, state='readonly')
        self.y_rows_stat_entry.grid(row=0, column=5, sticky="nsew")

        self.y_cols_stat_label = tk.Label(self.status_frame, text="y columns")
        self.y_cols_stat_label.grid(row=0, column=6, sticky="nsew")

        self.y_cols_stat_entry = tk.Entry(self.status_frame, textvariable=self.y_cols, state='readonly')
        self.y_cols_stat_entry.grid(row=0, column=7, sticky="nsew")

    def import_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("csv and xlsx Files", "*.csv *.xlsx")])
        if file_path:
            self.display_data(file_path)

    def display_data(self, file_path):

        if file_path.endswith('.csv'):
            self.df_raw = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            self.df_raw = pd.read_excel(file_path)

        self.display_table(self.raw_data_tree, self.df_raw)

    def display_table(self, tree, df):
        try:
            tree.delete(*tree.get_children())

        except _tkinter.TclError:
            pass

        tree['columns'] = df.columns.tolist()

        for col in tree['columns']:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", minwidth=80, width=80)

        for i, row in df.iterrows():
            formatted_row = []
            for value in row:
                if isinstance(value, float):
                    formatted_row.append(float("{:.4f}".format(value)))
                else:
                    formatted_row.append(value)
            tree.insert("", "end", values=formatted_row)

        self.activate_buttons()

    def open_select_features_window(self):
        if self.df_raw is not None:
            select_features_window = SelectColumnsWindow(self.master, self.df_raw)
            self.master.wait_window(select_features_window.window)
            self.selected_x_columns = select_features_window.selected_columns
            self.df_X = self.df_raw[self.selected_x_columns]
            self.display_table(self.features_data_tree, self.df_X)
            self.activate_buttons()
            self.x_rows.set(str(self.df_X.shape[0]))
            self.x_cols.set(str(self.df_X.shape[1]))
        else:
            pass

    def open_select_labels_window(self):
        if self.df_raw is not None:
            select_labels_window = SelectColumnsWindow(self.master, self.df_raw)
            self.master.wait_window(select_labels_window.window)
            self.selected_y_columns = select_labels_window.selected_columns
            self.df_y = self.df_raw[self.selected_y_columns]
            self.display_table(self.labels_data_tree, self.df_y)
            self.activate_buttons()
            self.y_rows.set(str(self.df_y.shape[0]))
            self.y_cols.set(str(self.df_y.shape[1]))
        else:
            pass

    def activate_buttons(self):
        if self.df_raw is not None or (self.df_X is None and self.df_y is None):
            self.select_y_button["state"] = "normal"
            self.select_x_button["state"] = "normal"
            if self.df_X is not None and self.df_y is not None:
                self.display_data_button["state"] = "normal"
                self.filter_data_button["state"] = "normal"
                self.standardscaler_button["state"] = "normal"
                self.msc_button["state"] = "normal"
                self.snv_button["state"] = "normal"
                self.pls_button["state"] = "normal"
                # self.optimize_button["state"] = "normal"
                self.pca_button["state"] = "normal"
                self.LDA_button["state"] = "normal"
                if self.df_y.shape[0] != self.df_X.shape[0]:
                    # self.select_y_button["state"] = "disabled"
                    # self.select_x_button["state"] = "disabled"
                    self.filter_data_button["state"] = "disabled"
                    self.standardscaler_button["state"] = "disabled"
                    self.msc_button["state"] = "disabled"
                    self.snv_button["state"] = "disabled"
                    self.pls_button["state"] = "disabled"
                    self.optimize_button["state"] = "disabled"
                    self.pca_button["state"] = "disabled"
                    self.LDA_button["state"] = "disabled"

    def display_xy(self):
        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.fig4.clear()

        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)
        ax3 = self.fig3.add_subplot(111)
        ax4 = self.fig4.add_subplot(111)

        ax1.plot(self.df_X.columns, self.df_X.T)
        ax1.set_title('X')
        ax1.set_xlabel('X columns [a.u]')
        ax1.set_ylabel('X values [a.u]')
        ax1.grid(True, alpha=0.3)
        # Specify tick positions manually
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

        for i in range(self.df_y.shape[1]):
            ax2.scatter(self.df_y.index, self.df_y.iloc[:, i], label=self.df_y.columns[i])
        ax2.set_title('y')
        ax2.set_xlabel('y columns [a.u]')
        ax2.set_ylabel('y values [a.u]')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

        dissimilarities = cdist(self.df_X, self.df_X, metric="correlation")
        normalized_dissimilarities_vector = (dissimilarities[1] - np.mean(dissimilarities[1])) / np.std(
            dissimilarities[1])

        ax3.scatter(self.df_X.index, normalized_dissimilarities_vector)
        ax3.set_title('Normalized Correlation Distance of X')
        ax3.set_xlabel('X rows [a.u]')
        ax3.set_ylabel('Amplitude [a.u]')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

        ax4.violinplot(self.df_y, showmeans=True, showmedians=False)
        ax4.boxplot(self.df_y, sym=".")
        ax4.set_xlabel('Value [a.u]')

        ax4.set_xticks(range(1, len(self.df_y.columns) + 1))
        ax4.set_xticklabels(self.df_y.columns)

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()

    def filter_data(self):

        self.select_y_button["state"] = "disabled"
        self.select_x_button["state"] = "disabled"

        filter_window = FilterData(self.master, self.df_X, self.df_y)

        self.master.wait_window(filter_window.window)

        self.df_X = filter_window.df_temp[self.selected_x_columns]
        self.df_y = filter_window.df_temp[self.selected_y_columns]

        self.display_table(self.features_data_tree, self.df_X)
        self.display_table(self.labels_data_tree, self.df_y)

        self.x_rows.set(str(self.df_X.shape[0]))
        self.x_cols.set(str(self.df_X.shape[1]))
        self.y_rows.set(str(self.df_y.shape[0]))
        self.y_cols.set(str(self.df_y.shape[1]))

    def apply_labels_normalization(self):
        """
        Apply Normalization
        """

        self.df_y = (self.df_y - np.mean(self.df_y, axis=0)) / np.std(self.df_y, axis=0)
        # self.df_X = (self.df_X - np.mean(self.df_X, axis=0)) / np.std(self.df_X, axis=0)
        # temp_input = self.df_X
        #
        # data_norm = temp_input.copy()
        # for i in range(temp_input.shape[0]):
        #     # Apply correction
        #     data_norm.iloc[i, :] = temp_input.iloc[i, :] / np.max(temp_input.iloc[i, :])
        #
        # self.df_X = data_norm.copy()
        #
        # self.display_table(self.features_data_tree, self.df_X)
        self.display_table(self.labels_data_tree, self.df_y)
        # self.display_table(self.features_data_tree, self.df_X)

    def apply_msc(self):
        """
        Apply Multiplicative Scatter Correction (MSC) to NIR spectra.
        """
        temp_input = self.df_X
        for i in range(temp_input.shape[0]):
            temp_input.loc[i, :] -= temp_input.loc[i, :].mean()

        ref = np.mean(temp_input, axis=0)

        # Define a new array and populate it with the corrected data
        data_msc = temp_input.copy()
        for i in range(temp_input.shape[0]):
            # Run regression
            fit = np.polyfit(ref, temp_input.iloc[i, :], 1, full=True)
            # Apply correction
            data_msc.iloc[i, :] = (temp_input.iloc[i, :] - fit[0][1]) / fit[0][0]

        self.df_X = data_msc.copy()

        self.display_table(self.features_data_tree, self.df_X)

    def apply_snv(self):
        """
        Apply Standard Normal Variate (SNV) correction to NIR spectra.
        """

        temp_input = self.df_X

        data_snv = temp_input.copy()
        for i in range(temp_input.shape[0]):
            # Apply correction
            data_snv.iloc[i, :] = (temp_input.iloc[i, :] -
                                   np.mean(temp_input.iloc[i, :])) / np.std(temp_input.iloc[i, :])

        self.df_X = data_snv.copy()

        self.display_table(self.features_data_tree, self.df_X)

    def open_pls_window(self):
        pls_window = PLS(self.master, self.df_X,
                         self.df_y)

        self.master.wait_window(pls_window.window)

        self.fig1.clear()

        ax1 = self.fig1.add_subplot(111)

        # Creating the legend table
        legend_table = ax1.table(cellText=[[f'Train', f'{pls_window.rmse_train:.6f}', f'{pls_window.r2_train:.6f}'],
                                           [f'CV', f'{pls_window.rmse_cv:.6f}', f'{pls_window.r2_cv:.6f}'],
                                           [f'Test', f'{pls_window.rmse_test:.6f}', f'{pls_window.r2_test:.6f}']],
                                 colLabels=['', 'RMSE', 'R-Square'],
                                 loc='upper left',
                                 cellLoc='center',
                                 cellColours=[['w', 'b', 'b'], ['w', 'r', 'r'], ['w', 'g', 'g']])

        # Styling the legend table
        legend_table.auto_set_font_size(False)
        legend_table.set_fontsize(10)
        legend_table.scale(0.4, 1.2)  # Adjust the size of the legend table

        z = np.polyfit(pls_window.y_train,
                       pls_window.y_pred_train, 1)

        ax1.scatter(pls_window.y_train, pls_window.y_pred_train,
                    color='blue', s=20, label="Train")
        ax1.scatter(pls_window.y_train, pls_window.y_pred_cv,
                    color='red', s=3, label="CV")
        ax1.scatter(pls_window.y_test, pls_window.y_pred_test,
                    color='green', s=3,
                    label='Test')
        ax1.plot(pls_window.y_train, pls_window.y_train,
                 color='k', linewidth=1, linestyle='--', label='Ideal Line')
        ax1.plot(np.polyval(z, pls_window.y_train), pls_window.y_train,
                 color='b', linewidth=1, linestyle='--', label='Model Line')
        ax1.set_title(f'Predicted vs True Results - Label={pls_window.selected_label} |'
                      f' PC#={pls_window.num_components} |'
                      f' Test/Train={pls_window.train_test_ratio * 100}%')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

        self.fig2.clear()
        ax2 = self.fig2.add_subplot(111)
        ax2.plot(self.df_X.columns, pls_window.pls.coef_[0, :])
        ax2.set_xlabel('X columns')
        ax2.set_ylabel('X Loadings')
        ax2.set_title('PLS Weights')
        # ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

        self.fig3.clear()

        x_scores = pls_window.pls.x_scores_
        y_scores = pls_window.pls.y_scores_

        ax3 = self.fig3.add_subplot(111)
        ax3.plot(x_scores[:, 0], x_scores[:, 1], 'ob', ms=4, label="X scores")
        ax3.plot(y_scores[:, 0], y_scores[:, 1], 'or', ms=4, label='y scores')

        ax3.set_xlabel('Component 1')
        ax3.set_ylabel('Component 2')
        ax3.set_title('Scores Plot')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

        self.fig4.clear()

        x_loadings = pls_window.pls.x_loadings_
        y_loadings = pls_window.pls.y_loadings_
        ax4 = self.fig4.add_subplot(111)
        ax4.plot(x_loadings[:, 0], x_loadings[:, 1], 'ob', ms=4, label='X loadings')
        ax4.plot(y_loadings[:, 0], y_loadings[:, 1], 'or', ms=4, label='y loadings')
        ax4.set_xlabel('Component 1')
        ax4.set_ylabel('Component 2')
        ax4.set_title('Loadings Plot')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()

        self.optimize_button["state"] = "normal"

    def optimize_pls_window(self):

        pls_opt_window = PLSOptimize(self.master, self.df_X,
                                     self.df_y)

        self.master.wait_window(pls_opt_window.window)

        # Find the index of the minimum RMSE score
        # min_rmse_index = np.argmin(pls_window.rmse_scores)
        #
        # ax2 = self.fig2.add_subplot(111)
        # ax2.plot(pls_window.components_range, pls_window.rmse_scores, marker='o')
        # ax2.plot(pls_window.components_range[min_rmse_index], pls_window.rmse_scores[min_rmse_index],
        #          'P', ms=6, mfc='red',
        #          label=f'Optimized Number of Components={pls_window.components_range[min_rmse_index]}')
        # ax2.set_xlabel('Number of Components')
        # ax2.set_ylabel('RMSE')
        # ax2.set_title('RMSE vs Number of Components')
        # ax2.legend(loc='best')
        # ax2.grid(True)

    def open_classification_window(self, class_type):
        classification_window = Classification(self.master, self.df_X,
                                               self.df_y, class_type)

        self.master.wait_window(classification_window.window)

        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.fig4.clear()

        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)
        ax3 = self.fig3.add_subplot(111)
        ax4 = self.fig4.add_subplot(111)

        if classification_window.class_type == 'PCA':
            scatter = ax1.scatter(classification_window.x_pca[:, classification_window.x_pc],
                                  classification_window.x_pca[:, classification_window.y_pc],
                                  c=self.df_y[classification_window.selected_label].values)
            ax1.set_title('PCA')
            ax1.set_xlabel('PC' + str(classification_window.x_pc + 1))
            ax1.set_ylabel('PC' + str(classification_window.y_pc + 1))
            ax1.grid(True, alpha=0.3)
            # Specify tick positions manually
            ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

            cbar = self.fig1.colorbar(scatter)
            cbar.set_label(classification_window.selected_label)

            ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
                     classification_window.explained_variance_ratio, '-ob',
                     label='Explained Variance ratio')
            ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
                     np.cumsum(classification_window.explained_variance_ratio), '-or',
                     label='Cumulative Variance ratio')
            ax2.set_title('Explained Variance')
            ax2.set_xlabel('PC Number')
            ax2.set_ylabel('Ratio')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
            ax2.set_xlim(0.5, 10.5)
            ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

            cmap = plt.get_cmap('viridis')
            for i, j in enumerate(classification_window.eucl_dist):
                arrow_col = (classification_window.eucl_dist[i] - np.array(classification_window.eucl_dist).min()) / (
                        np.array(classification_window.eucl_dist).max() -
                        np.array(classification_window.eucl_dist).min())
                ax3.arrow(0, 0,  # Arrows start at the origin
                          classification_window.ccircle[i][0],  # 0 for PC1
                          classification_window.ccircle[i][1],  # 1 for PC2
                          lw=2,  # line width
                          length_includes_head=True,
                          color=cmap(arrow_col),
                          fc=cmap(arrow_col),
                          head_width=0.05,
                          head_length=0.05)
                ax3.text((classification_window.ccircle[i][0]), (classification_window.ccircle[i][1]),
                         self.df_X.columns[i], size=8)
            # Draw the unit circle, for clarity
            circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
            ax3.add_patch(circle)
            ax3.set_title('PCA Correlation Circle')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

            arr_2d = np.array(classification_window.eucl_dist).reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(arr_2d)

            eucl_dist_scaled = scaler.transform(arr_2d).flatten()
            for i in range(self.df_X.shape[1]):
                ax4.scatter(float(self.df_X.columns.values[i]), self.df_X.mean(axis=0).values[i],
                            color=cmap(eucl_dist_scaled[i]))
            ax4.set_title('Correlation Bands')
            ax4.set_xlabel('X columns')
            ax4.set_ylabel('Value')

            ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax4.grid(True, alpha=0.3)

        elif classification_window.class_type == 'LDA':

            scatter = ax1.scatter(classification_window.x_lda[:, classification_window.x_pc],
                                  classification_window.x_lda[:, classification_window.y_pc],
                                  c=classification_window.y_binned)
            ax1.set_title('LDA')
            ax1.set_xlabel('PC' + str(classification_window.x_pc + 1))
            ax1.set_ylabel('PC' + str(classification_window.y_pc + 1))
            ax1.grid(True, alpha=0.3)
            # Specify tick positions manually
            ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

            cbar = self.fig1.colorbar(scatter)
            cbar.set_label(classification_window.selected_label + ' (binned)')

            ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
                     classification_window.explained_variance_ratio, '-ob',
                     label='Explained Variance ratio')
            ax2.plot(range(1, len(classification_window.explained_variance_ratio) + 1),
                     np.cumsum(classification_window.explained_variance_ratio), '-or',
                     label='Cumulative Variance ratio')
            ax2.set_title('Explained Variance')
            ax2.set_xlabel('PC Number')
            ax2.set_ylabel('Ratio')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
            ax2.set_xlim(0.5, 10.5)
            ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

            cmap = plt.get_cmap('viridis')
            for i, j in enumerate(classification_window.eucl_dist):
                arrow_col = (classification_window.eucl_dist[i] - np.array(classification_window.eucl_dist).min()) / (
                        np.array(classification_window.eucl_dist).max() -
                        np.array(classification_window.eucl_dist).min())
                ax3.arrow(0, 0,  # Arrows start at the origin
                          classification_window.ccircle[i][0],  # 0 for PC1
                          classification_window.ccircle[i][1],  # 1 for PC2
                          lw=2,  # line width
                          length_includes_head=True,
                          color=cmap(arrow_col),
                          fc=cmap(arrow_col),
                          head_width=0.05,
                          head_length=0.05)
                ax3.text((classification_window.ccircle[i][0]), (classification_window.ccircle[i][1]),
                         self.df_X.columns[i], size=8)
            # Draw the unit circle, for clarity
            circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
            ax3.add_patch(circle)
            ax3.set_title('LDA Correlation Circle')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

            arr_2d = np.array(classification_window.eucl_dist).reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(arr_2d)

            eucl_dist_scaled = scaler.transform(arr_2d).flatten()
            for i in range(self.df_X.shape[1]):
                ax4.scatter(float(self.df_X.columns.values[i]), self.df_X.mean(axis=0).values[i],
                            color=cmap(eucl_dist_scaled[i]))
            ax4.set_title('Correlation Bands')
            ax4.set_xlabel('X columns')
            ax4.set_ylabel('Value')

            ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax4.grid(True, alpha=0.3)

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()


def main():
    root = tk.Tk()

    # Import the tcl file with the tk.call method
    # root.tk.call('source', os.getcwd() + '/Styles/Azure-ttk-theme/azure.tcl')
    root.tk.call('source', os.getcwd() + '/Styles/ttk-Breeze/breeze.tcl')
    # root.tk.call('source', os.getcwd() + '/Styles/Forest-ttk-theme/forest-light.tcl')
    # root.tk.call('source', os.getcwd() + '/Styles/Forest-ttk-theme/forest-dark.tcl')

    app = MyChemometrix(root)
    root.mainloop()


if __name__ == "__main__":
    main()

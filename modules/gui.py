from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import _tkinter
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
from modules.preprocessing import DataInspection, DataFiltering, DataStandardization
from modules.regression import PLS
from modules.classification import Classification
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.patches import Circle
from sklearn.preprocessing import MinMaxScaler


class Main:
    def __init__(self, root):

        self.root = root
        self.root.title("myChemometriX")

        # Get screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.root.geometry(f'{screen_width}x{screen_height}')

        # Calculate frame sizes
        buttons_frame_height = screen_height // 25
        status_frame_height = screen_height // 25
        data_frame_width = screen_width * 0.25
        graphs_frame_width = screen_width * 0.75

        # Styles
        self.style = ttk.Style()
        self.style.theme_use("Breeze")

        self.style.configure("TButton", background="light blue", foreground="black", padding=5)
        self.style.map("TButton", background=[("active", "light sky blue")])

        # Frames
        self.buttons_frame = ttk.Frame(root, height=buttons_frame_height)
        self.buttons_frame.pack(side="top", fill="x")

        self.status_frame = ttk.Frame(root, height=status_frame_height)
        self.status_frame.pack(side="bottom", fill="x")

        # PanedWindow for resizable frames
        self.paned_window = ttk.PanedWindow(root, orient="horizontal")
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

        self.test_model_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Test Model")
        self.test_model_buttons_frame.pack(side="left", fill="both", expand=True)

        self.export_buttons_frame = ttk.LabelFrame(self.buttons_frame, text="Export Results")
        self.export_buttons_frame.pack(side="left", fill="both", expand=True)

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
                                              command=lambda: self.display_xy(self.df_x, self.df_y))
        self.display_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.filter_data_button = ttk.Button(self.preprocessing_buttons_frame, text="Filter Data", state="disabled",
                                             command=self.open_data_filtering_window)
        self.filter_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.augment_data_button = ttk.Button(self.preprocessing_buttons_frame, text="Augment Data", state="disabled",
                                              command=self.apply_augmentation)
        self.augment_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.standardscaler_button = ttk.Button(self.preprocessing_buttons_frame, text="Label Normalization",
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
                                     command=self.open_regression_window)
        self.pls_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pca_button = ttk.Button(self.classifier_buttons_frame, text="PCA", state="disabled",
                                     command=lambda: self.open_classification_window('PCA'))
        self.pca_button.pack(side="left", fill="both", padx=5, pady=5)

        self.LDA_button = ttk.Button(self.classifier_buttons_frame, text="LDA", state="disabled",
                                     command=lambda: self.open_classification_window('LDA'))
        self.LDA_button.pack(side="left", fill="both", padx=5, pady=5)

        self.load_test_button = ttk.Button(self.test_model_buttons_frame, text="Load Test", state="disabled",
                                           command=False)
        self.load_test_button.pack(side="left", fill="both", padx=5, pady=5)

        self.export_data_button = ttk.Button(self.export_buttons_frame, text="Export", state="disabled",
                                             command=False)
        self.export_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.df_raw = None
        self.df_x = None
        self.df_y = None
        self.selected_x_columns = []
        self.selected_y_columns = []

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
            self.read_to_dataframe(file_path)

    def read_to_dataframe(self, file_path):

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
            select_features_window = DataSelectionWin(self.root, self.df_raw)
            self.root.wait_window(select_features_window.window)
            self.selected_x_columns = select_features_window.selected_columns
            self.df_x = self.df_raw[self.selected_x_columns]
            # self.df_x = DataStandardization(self.df_x, self.df_y).apply_msc()
            # self.df_x = DataStandardization(self.df_x, self.df_y).area_normalization()
            self.display_table(self.features_data_tree, self.df_x)
            self.activate_buttons()
            self.x_rows.set(str(self.df_x.shape[0]))
            self.x_cols.set(str(self.df_x.shape[1]))
        else:
            pass

    def open_select_labels_window(self):
        if self.df_raw is not None:
            select_labels_window = DataSelectionWin(self.root, self.df_raw)
            self.root.wait_window(select_labels_window.window)
            self.selected_y_columns = select_labels_window.selected_columns
            self.df_y = self.df_raw[self.selected_y_columns]
            self.display_table(self.labels_data_tree, self.df_y)
            self.activate_buttons()
            self.y_rows.set(str(self.df_y.shape[0]))
            self.y_cols.set(str(self.df_y.shape[1]))
        else:
            pass

    def activate_buttons(self):
        if self.df_raw is not None or (self.df_x is None and self.df_y is None):
            self.select_y_button["state"] = "normal"
            self.select_x_button["state"] = "normal"
            if self.df_x is not None and self.df_y is not None:
                self.display_data_button["state"] = "normal"
                self.augment_data_button["state"] = "normal"
                self.filter_data_button["state"] = "normal"
                self.standardscaler_button["state"] = "normal"
                self.msc_button["state"] = "normal"
                self.snv_button["state"] = "normal"
                self.pls_button["state"] = "normal"
                # self.optimize_button["state"] = "normal"
                self.pca_button["state"] = "normal"
                self.LDA_button["state"] = "normal"
                if self.df_y.shape[0] != self.df_x.shape[0]:
                    # self.select_y_button["state"] = "disabled"
                    # self.select_x_button["state"] = "disabled"
                    self.filter_data_button["state"] = "disabled"
                    self.standardscaler_button["state"] = "disabled"
                    self.msc_button["state"] = "disabled"
                    self.snv_button["state"] = "disabled"
                    self.pls_button["state"] = "disabled"
                    self.pca_button["state"] = "disabled"
                    self.LDA_button["state"] = "disabled"

    def display_xy(self, df_x, df_y):

        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.fig4.clear()

        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)
        ax3 = self.fig3.add_subplot(111)
        ax4 = self.fig4.add_subplot(111)

        ax1.plot(df_x.columns, df_x.T)
        ax1.set_title('X')
        ax1.set_xlabel('X columns [a.u]')
        ax1.set_ylabel('X values [a.u]')
        ax1.grid(True, alpha=0.3)
        # Specify tick positions manually
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

        for i in range(df_y.shape[1]):
            ax2.scatter(df_y.index, df_y.iloc[:, i], label=df_y.columns[i])
        ax2.set_title('y')
        ax2.set_xlabel('y columns [a.u]')
        ax2.set_ylabel('y values [a.u]')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

        dissimilarities = DataInspection(df_x, df_y).anomaly_detection()

        ax3.scatter(df_x.index, dissimilarities)
        ax3.set_title('Absolute Normalized Correlation Distance of X')
        ax3.set_xlabel('X rows [a.u]')
        ax3.set_ylabel('Amplitude [a.u]')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

        ax4.violinplot(df_y, showmeans=True, showmedians=False)
        ax4.boxplot(df_y, sym=".")
        ax4.set_xlabel('Value [a.u]')

        ax4.set_xticks(range(1, len(df_y.columns) + 1))
        ax4.set_xticklabels(df_y.columns)

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()

    def open_data_filtering_window(self):

        self.select_y_button["state"] = "disabled"
        self.select_x_button["state"] = "disabled"

        filter_window = DataFilteringWin(self.root, self.df_x, self.df_y)

        self.root.wait_window(filter_window.window)

        self.df_x = filter_window.df_x
        self.df_y = filter_window.df_y

        self.display_table(self.features_data_tree, self.df_x)
        self.display_table(self.labels_data_tree, self.df_y)

        self.x_rows.set(str(self.df_x.shape[0]))
        self.x_cols.set(str(self.df_x.shape[1]))
        self.y_rows.set(str(self.df_y.shape[0]))
        self.y_cols.set(str(self.df_y.shape[1]))

        self.display_xy(self.df_x, self.df_y)

    def apply_augmentation(self):

        df_temp = pd.concat([self.df_y, self.df_x], axis=1)
        # Group the DataFrame according to the specific column
        grouped = df_temp.groupby('test #')

        flag = 'max'
        equalized_data = []

        if flag == 'min':

            # Determine the size of the smallest group
            min_group_size = grouped.size().min()

            # Sample data from each group to make them the same size

            for _, group_data in grouped:
                equalized_data.append(group_data.sample(n=min_group_size, random_state=None))

        elif flag == 'max':

            # Determine the size of the largest group
            max_group_size = grouped.size().max()

            noise_gain = 0.4

            for name, group_data in grouped:
                group_size = len(group_data)

                # Append the original group data
                equalized_data.append(group_data)

                # Calculate the number of rows to pad
                rows_to_add = max_group_size - group_size

                if rows_to_add > 0:
                    # Calculate the mean and std values for the group's columns
                    mean_values = group_data.mean()
                    std_values = noise_gain * group_data.std()

                    # Create new rows with mean + random value between -std and +std
                    padding_data = {col: np.random.uniform(mean - std, mean + std, rows_to_add)
                                    for col, mean, std in zip(group_data.columns, mean_values, std_values)}
                    padding_df = pd.DataFrame(padding_data)

                    # Set the group column to the current group's name
                    padding_df['test #'] = name

                    # Append the padding data
                    equalized_data.append(padding_df)

        # Concatenate the original and padded data into a new DataFrame
        equalized_df = pd.concat(equalized_data, ignore_index=True)

        self.df_x = equalized_df[self.selected_x_columns]
        self.df_y = equalized_df[self.selected_y_columns]

        self.display_table(self.features_data_tree, self.df_x)
        self.display_table(self.labels_data_tree, self.df_y)

        self.x_rows.set(str(self.df_x.shape[0]))
        self.x_cols.set(str(self.df_x.shape[1]))
        self.y_rows.set(str(self.df_y.shape[0]))
        self.y_cols.set(str(self.df_y.shape[1]))

        self.display_xy(self.df_x, self.df_y)

    def apply_labels_normalization(self):
        df_y = DataStandardization(self.df_x, self.df_y).apply_labels_normalization()
        self.display_table(self.labels_data_tree, self.df_y)
        self.display_xy(self.df_x, df_y)

    def apply_msc(self):
        self.df_x = DataStandardization(self.df_x, self.df_y).apply_msc()
        self.display_table(self.features_data_tree, self.df_x)
        self.display_xy(self.df_x, self.df_y)

    def apply_snv(self):
        self.df_x = DataStandardization(self.df_x, self.df_y).apply_snv()
        self.display_table(self.features_data_tree, self.df_x)
        self.display_xy(self.df_x, self.df_y)

    def open_regression_window(self):
        regression_window = RegressionWin(self.root, self.df_x, self.df_y)

        self.root.wait_window(regression_window.window)

        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.fig4.clear()

        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)
        ax3 = self.fig3.add_subplot(111)
        ax4 = self.fig4.add_subplot(111)

        label = regression_window.label_to_display
        num_components = regression_window.pls.best_estimator_.n_components

        y_train = regression_window.y_train[label]

        y_pred_train = regression_window.y_pred_train[label]
        y_pred_cv = regression_window.y_pred_cv[label]

        rmse_train = regression_window.rmse_train
        rmse_cv = regression_window.rmse_cv

        r2_train = regression_window.r2_train
        r2_cv = regression_window.r2_cv

        z_train = np.polyfit(y_train.values, y_pred_train.values, 1)
        z_cv = np.polyfit(y_train.values, y_pred_cv.values, 1)

        if not regression_window.y_test.empty:
            y_test = regression_window.y_test[label]
            y_pred_test = regression_window.y_pred_test[label]
            rmse_test = regression_window.rmse_test
            r2_test = regression_window.r2_test
            z_test = np.polyfit(y_test.values, y_pred_test.values, 1)
        else:
            y_test = pd.DataFrame([])
            y_pred_test = pd.DataFrame([])
            rmse_test = 0
            r2_test = 0
            z_test = [0, 0]

        # Creating the legend table
        legend_table = ax1.table(
            cellText=[[f'Train', f'{rmse_train:.6f}', f'{r2_train:.6f}', f'{z_train[0]:.6f}'],
                      [f'CV', f'{rmse_cv:.6f}', f'{r2_cv:.6f}', f'{z_cv[0]:.6f}'],
                      [f'Test', f'{rmse_test:.6f}', f'{r2_test:.6f}', f'{z_test[0]:.6f}']],
            colLabels=['', 'RMSE', 'R-Square', 'Slope'],
            loc='upper left',
            cellLoc='center',
            cellColours=[['w', 'b', 'b', 'b'], ['w', 'r', 'r', 'r'], ['w', 'g', 'g', 'g']])

        # Styling the legend table
        legend_table.auto_set_font_size(False)
        legend_table.set_fontsize(10)
        legend_table.scale(0.6, 1.2)  # Adjust the size of the legend table

        z = np.polyfit(y_train, y_pred_train, 1)

        ax1.scatter(y_train, y_pred_train,
                    color='blue', s=20, label="Train")
        ax1.scatter(y_train, y_pred_cv,
                    color='red', s=3, label="CV")
        if y_pred_test is not []:
            ax1.scatter(y_test, y_pred_test,
                        color='green', s=3,
                        label='Test')
        ax1.plot(y_train, y_train,
                 color='k', linewidth=1, linestyle='--', label='Ideal Line')
        ax1.plot(np.polyval(z, y_train),
                 y_train,
                 color='b', linewidth=1, linestyle='--', label='Model Line')
        ax1.set_title(f'Predicted vs True Results - Label={label} |'
                      f' PC#={num_components} |'
                      f' Test/Train={regression_window.test_ratio * 100}%')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

        i = 0
        for i, _label in enumerate(regression_window.selected_labels):
            if label == _label:
                break

        ax2.plot(self.df_x.columns, regression_window.pls.best_estimator_.coef_[i], label=label)
        ax2.set_xlabel('X columns')
        ax2.set_ylabel('X Loadings')
        ax2.set_title('Regression Coefficients')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))

        x_scores = regression_window.pls.best_estimator_.x_scores_
        y_scores = regression_window.pls.best_estimator_.y_scores_

        ax3.plot(x_scores[:, 0], -x_scores[:, 1], 'o', ms=4, label=f"X scores")
        ax3.plot(y_scores[:, 0], y_scores[:, 1], 'o', ms=4, label=f'y scores')

        ax3.set_xlabel('Component 1')
        ax3.set_ylabel('Component 2')
        ax3.set_title('Scores Plot')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

        x_loadings = regression_window.pls.best_estimator_.x_loadings_
        y_loadings = regression_window.pls.best_estimator_.y_loadings_

        x_std = 1
        y_std = 1
        if len(x_loadings) > 1:
            x_std = np.std(x_loadings, axis=0)
        if len(y_loadings) > 1:
            y_std = np.std(y_loadings, axis=0)

        x_loadings = (x_loadings - np.mean(x_loadings, axis=0)) / x_std
        y_loadings = (y_loadings - np.mean(y_loadings, axis=0)) / y_std
        ax4.plot(x_loadings[:, 0], x_loadings[:, 1], 'o', ms=4, label=f'X loadings')
        ax4.plot(y_loadings[:, 0], y_loadings[:, 1], 'o', ms=4, label=f'y loadings')
        for i, feature in enumerate(self.df_x.columns):
            ax4.annotate(feature, (x_loadings[i, 0], x_loadings[i, 1]), textcoords="offset points",
                         xytext=(5, 5), fontsize=6, ha='right')
        for i, label in enumerate(regression_window.selected_labels):
            ax4.annotate(label, (y_loadings[i, 0], y_loadings[i, 1]), textcoords="offset points",
                         xytext=(5, 5), fontsize=8, ha='right')
        ax4.set_xlabel('Component 1')
        ax4.set_ylabel('Component 2')
        ax4.set_title('Loadings Plot')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # elif regression_window.opt_flag:
        #
        #     rmse_min_index = regression_window.rmsemin
        #
        #     ax3.plot(regression_window.components_range, regression_window.rmse_cv_per_component, marker='o')
        #     ax3.plot(regression_window.components_range[rmse_min_index],
        #              regression_window.rmse_cv_per_component[rmse_min_index],
        #              'P', ms=6, mfc='red',
        #              label=f'Optimized Number of'
        #                    f' Components={regression_window.components_range[rmse_min_index]}')
        #     ax3.set_xlabel('Number of Components')
        #     ax3.set_ylabel('RMSE')
        #     ax3.set_title('RMSE vs Number of Components')
        #     ax3.legend(loc='best')
        #     ax3.grid(True)
        #     ax3.xaxis.set_major_locator(plt.MaxNLocator(20))
        #
        #     self.fig4.clear()
        #     ax4 = self.fig4.add_subplot(111)
        #     ax4.plot(regression_window.t_squared, regression_window.q_residuals, 'o')
        #     ax4.plot([regression_window.t_squared_conf, regression_window.t_squared_conf],
        #              [ax4.axis()[2], ax4.axis()[3]], '--')
        #     ax4.plot([ax4.axis()[0], ax4.axis()[1]], [regression_window.q_residuals_conf,
        #                                               regression_window.q_residuals_conf], '--')
        #     ax4.set_title('Outliers Map')
        #     ax4.set_xlabel("Hotelling's T-squared")
        #     ax4.set_ylabel('Q residuals')
        #
        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()

        regression_window.show_prediction_results()

    def open_classification_window(self, class_type):
        classification_window = ClassificationWin(self.root, self.df_x, self.df_y, class_type)

        self.root.wait_window(classification_window.window)

        self.fig1.clear()
        self.fig2.clear()
        self.fig3.clear()
        self.fig4.clear()

        ax1 = self.fig1.add_subplot(111)
        ax2 = self.fig2.add_subplot(111)
        ax3 = self.fig3.add_subplot(111)
        ax4 = self.fig4.add_subplot(111)

        if class_type == 'PCA':
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
            for i, j in enumerate(classification_window.euclidian_dist):
                arrow_col = (classification_window.euclidian_dist[i] - np.array(classification_window.euclidian_dist).min()) / (
                        np.array(classification_window.euclidian_dist).max() -
                        np.array(classification_window.euclidian_dist).min())
                ax3.arrow(0, 0,  # Arrows start at the origin
                          classification_window.corr_circle[i][0],  # 0 for PC1
                          classification_window.corr_circle[i][1],  # 1 for PC2
                          lw=2,  # line width
                          length_includes_head=True,
                          color=cmap(arrow_col),
                          fc=cmap(arrow_col),
                          head_width=0.05,
                          head_length=0.05)
                ax3.text((classification_window.corr_circle[i][0]), (classification_window.corr_circle[i][1]),
                         self.df_x.columns[i], size=8)
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

            arr_2d = np.array(classification_window.euclidian_dist).reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(arr_2d)

            eucl_dist_scaled = scaler.transform(arr_2d).flatten()
            for i in range(self.df_x.shape[1]):
                ax4.scatter(float(self.df_x.columns.values[i]), self.df_x.mean(axis=0).values[i],
                            color=cmap(eucl_dist_scaled[i]))
            ax4.set_title('Correlation Bands')
            ax4.set_xlabel('X columns')
            ax4.set_ylabel('Value')

            ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax4.grid(True, alpha=0.3)

        elif classification_window.class_type == 'LDA':
            cluster_means = []
            for label in np.unique(classification_window.y_binned):
                cluster_means.append(np.mean(classification_window.x_lda[classification_window.y_binned == label],
                                             axis=0))

            scatter = ax1.scatter(classification_window.x_lda[:, classification_window.x_pc],
                                  classification_window.x_lda[:, classification_window.y_pc],
                                  c=classification_window.y_binned, cmap='viridis', alpha=0.5)
            ax1.scatter(np.array(cluster_means)[:, classification_window.x_pc],
                        np.array(cluster_means)[:, classification_window.y_pc],
                        c='red', marker='x', label='Cluster Means')
            ax1.set_title('LDA')
            ax1.set_xlabel('PC' + str(classification_window.x_pc + 1))
            ax1.set_ylabel('PC' + str(classification_window.y_pc + 1))
            ax1.legend(loc='best')
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
            for i, j in enumerate(classification_window.euclidian_dist):
                arrow_col = (classification_window.euclidian_dist[i] - np.array(classification_window.euclidian_dist).min()) / (
                        np.array(classification_window.euclidian_dist).max() -
                        np.array(classification_window.euclidian_dist).min())
                ax3.arrow(0, 0,  # Arrows start at the origin
                          classification_window.corr_circle[i][0],  # 0 for PC1
                          classification_window.corr_circle[i][1],  # 1 for PC2
                          lw=2,  # line width
                          length_includes_head=True,
                          color=cmap(arrow_col),
                          fc=cmap(arrow_col),
                          head_width=0.05,
                          head_length=0.05)
                ax3.text((classification_window.corr_circle[i][0]), (classification_window.corr_circle[i][1]),
                         self.df_x.columns[i], size=8)
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

            arr_2d = np.array(classification_window.euclidian_dist).reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(arr_2d)

            eucl_dist_scaled = scaler.transform(arr_2d).flatten()
            for i in range(self.df_x.shape[1]):
                ax4.scatter(float(self.df_x.columns.values[i]), self.df_x.mean(axis=0).values[i],
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


class DataSelectionWin:
    def __init__(self, root, df):
        self.root = root
        self.columns = df.select_dtypes(include=[np.number]).columns
        self.selected_columns = []

        self.window = tk.Toplevel(root)
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


class DataFilteringWin:
    def __init__(self, root, df_x, df_y):
        self.root = root
        self.df_x = df_x
        self.df_y = df_y

        self.window = tk.Toplevel(root)
        self.window.title("Data Filtering")
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

        self.filter_button = ttk.Button(self.window, text="Filter", command=self.filter_data)
        self.filter_button.grid(row=3, column=0, padx=5, pady=10, sticky='ew')

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=3, column=1, padx=5, pady=10, sticky='ew')

    def filter_data(self):
        x_threshold = float(self.x_std_num_entry.get())
        y_threshold = float(self.y_std_num_entry.get())
        self.df_x, self.df_y = DataFiltering(self.df_x, self.df_y).filter_data(x_threshold, y_threshold)

        self.window.destroy()


class ClassificationWin:
    def __init__(self, root, df_x, df_y, class_type):
        self.y_binned = None
        self.x_lda = None
        self.lda = None
        self.selected_label = ''
        self.class_type = class_type
        self.x_pc = 0
        self.y_pc = 1
        self.euclidian_dist = []
        self.corr_circle = []
        self.explained_variance_ratio = []
        self.x_pca = []
        self.pca = []
        self.root = root
        self.df_x = df_x
        self.df_y = df_y
        self.num_components = []

        self.window = tk.Toplevel(root)
        self.window.title("Classification Options")
        # self.window.geometry("360x200")

        # Label and entry for number of components
        self.x_pc_label = ttk.Label(self.window, text="x-axis component:")
        self.x_pc_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.x_pc_entry = ttk.Entry(self.window)
        self.x_pc_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.x_pc_entry.insert(tk.END, "1")

        # Label and entry for number of components
        self.y_pc_label = ttk.Label(self.window, text="y-axis component:")
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
        self.apply_button = ttk.Button(self.window, text="Apply",
                                       command=self.apply_classification)
        self.apply_button.grid(row=3, column=0, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=3, column=1, padx=5, pady=5)

    def apply_classification(self):
        self.selected_label = self.select_label_combobox.get()
        self.x_pc = int(self.x_pc_entry.get()) - 1
        self.y_pc = int(self.y_pc_entry.get()) - 1
        self.num_components = max([self.x_pc, self.y_pc])

        if self.class_type == 'PCA':

            self.pca, self.x_pca, self.explained_variance_ratio, self.corr_circle, self.euclidian_dist = (
                Classification(self.df_x, self.df_y).apply_classification(self.class_type, self.num_components,
                                                                          self.selected_label))
        elif self.class_type == 'LDA':
            (self.lda, self.x_lda, self.explained_variance_ratio, self.corr_circle,
             self.euclidian_dist, self.y_binned) = (
                Classification(self.df_x, self.df_y).apply_classification(self.class_type, self.num_components,
                                                                          self.selected_label))

        self.window.destroy()


class RegressionWin:
    def __init__(self, root, df_x, df_y):

        self.root = root
        self.df_x = df_x
        self.df_y = df_y
        self.selected_labels = []
        self.label_to_display = []
        self.group_by = []
        self.num_components = []
        self.test_ratio = []
        self.components_range = np.arange(1, 20)
        self.pls = None
        self.y_train = pd.DataFrame([])
        self.y_test = pd.DataFrame([])
        self.y_pred_train = pd.DataFrame([])
        self.y_pred_cv = pd.DataFrame([])
        self.y_pred_test = pd.DataFrame([])
        self.train_groups = pd.DataFrame([])
        self.test_groups = pd.DataFrame([])
        self.r2_train = []
        self.r2_cv = []
        self.r2_test = []
        self.rmse_train = []
        self.rmse_cv = []
        self.rmse_test = []
        self.rmse_cv_per_component = []
        self.rmsemin = 1
        self.conf = 0.95
        self.max_outliers = 20

        self.window = tk.Toplevel(root)
        self.window.title("PLS Options")
        # self.window.geometry("360x200")

        self.select_label_label = ttk.Label(self.window, text="Choose y labels:")
        self.select_label_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.selected_labels_entry = ttk.Entry(self.window)
        self.selected_labels_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.select_y_button = ttk.Button(self.window, text="Select labels",
                                          command=self.open_select_labels_window)
        self.select_y_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

        # ComboBox for selecting label
        self.select_label_label = ttk.Label(self.window, text="Label to Display:")
        self.select_label_label.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.select_label_combobox = ttk.Combobox(self.window, values=[])
        self.select_label_combobox.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        self.num_components_label = ttk.Label(self.window, text="Maximum Number of Components:")
        self.num_components_label.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        self.num_components_entry = ttk.Entry(self.window)
        self.num_components_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        self.num_components_entry.insert(tk.END, "8")

        self.test_ratio_label = ttk.Label(self.window, text="Test/Train Ratio (0-0.5):")
        self.test_ratio_label.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

        self.test_ratio_entry = ttk.Entry(self.window)
        self.test_ratio_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        self.test_ratio_entry.insert(tk.END, "0.0")

        # ComboBox for selecting label
        self.group_by_label = ttk.Label(self.window, text="Group By:")
        self.group_by_label.grid(row=4, column=0, padx=5, pady=5, sticky='ew')

        self.group_by_combobox = ttk.Combobox(self.window, values=[])
        self.group_by_combobox.grid(row=4, column=1, padx=5, pady=5, sticky='ew')

        self.num_outliers_label = ttk.Label(self.window, text="Maximum Number of Outliers to Remove:")
        self.num_outliers_label.grid(row=5, column=0, padx=5, pady=5, sticky='ew')

        self.num_outliers_entry = ttk.Entry(self.window)
        self.num_outliers_entry.grid(row=5, column=1, padx=5, pady=5, sticky='ew')
        self.num_outliers_entry.insert(tk.END, "0")

        self.confidence_label = ttk.Label(self.window, text="Confidence Level (0.5-0.99):")
        self.confidence_label.grid(row=6, column=0, padx=5, pady=5, sticky='ew')

        self.confidence_entry = ttk.Entry(self.window)
        self.confidence_entry.grid(row=6, column=1, padx=5, pady=5, sticky='ew')
        self.confidence_entry.insert(tk.END, "0.95")

        # Apply and Cancel buttons
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_pls)
        self.apply_button.grid(row=7, column=0, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=7, column=1, padx=5, pady=5)

    def open_select_labels_window(self):
        select_labels_window = DataSelectionWin(self.root, self.df_y)
        self.root.wait_window(select_labels_window.window)
        self.selected_labels = select_labels_window.selected_columns
        self.selected_labels_entry.delete(0, tk.END)
        self.selected_labels_entry.insert(tk.END, str(self.selected_labels))
        self.select_label_combobox['values'] = self.selected_labels
        self.group_by_combobox['values'] = self.df_y.columns.tolist()
        self.select_label_combobox.current(0)
        value = 'test #'
        try:
            index = self.group_by_combobox['values'].index(value)
            self.group_by_combobox.current(index)
        except ValueError:
            self.group_by_combobox.current(0)

    def apply_pls(self):

        self.num_components = min(int(self.num_components_entry.get()),
                                  self.df_x.shape[0], self.df_x.shape[1])
        self.label_to_display = str(self.select_label_combobox.get())
        self.group_by = str(self.group_by_combobox.get())
        self.max_outliers = str(self.num_outliers_entry.get())
        self.conf = str(self.confidence_entry.get())

        self.test_ratio = float(self.test_ratio_entry.get())
        if self.test_ratio < 0:
            self.test_ratio = 0
        elif self.test_ratio > 0.5:
            self.test_ratio = 0.5

        (self.pls, self.y_train, self.y_test, self.y_pred_train,
         self.y_pred_cv, self.y_pred_test, self.train_groups, self.test_groups) = \
            PLS(self.df_x, self.df_y).apply_pls(self.num_components,
                                                self.selected_labels,
                                                self.test_ratio,
                                                self.label_to_display,
                                                self.group_by,
                                                int(self.max_outliers),
                                                float(self.conf))

        self.rmse_train, self.r2_train = (np.sqrt(
            mean_squared_error(self.y_train[self.label_to_display], self.y_pred_train[self.label_to_display])),
                                          r2_score(self.y_train[self.label_to_display],
                                                   self.y_pred_train[self.label_to_display]))

        self.rmse_cv, self.r2_cv = (np.sqrt(mean_squared_error(self.y_train[self.label_to_display],
                                                               self.y_pred_cv[self.label_to_display])),
                                    r2_score(self.y_train[self.label_to_display],
                                             self.y_pred_cv[self.label_to_display]))

        if not self.y_pred_test.empty:

            self.rmse_test, self.r2_test = (np.sqrt(
                mean_squared_error(self.y_test[self.label_to_display], self.y_pred_test[self.label_to_display])),
                                            r2_score(self.y_test[self.label_to_display],
                                                     self.y_pred_test[self.label_to_display]))

        else:
            pass

        self.window.destroy()

        dict_ = {'Ref': [],
                 'Train/Test': [],
                 'Mean': [],
                 'STD': [],
                 'MAE': [],
                 'Relative Error %': []}

        results_df = pd.DataFrame(dict_)

        for indx, i in enumerate(np.unique(self.train_groups)):
            ref = self.y_train[self.label_to_display][self.train_groups.values == i].values.mean()
            class_ = 'Train'
            mean = self.y_pred_train[self.label_to_display][self.train_groups.values == i].values.mean()
            std = self.y_pred_train[self.label_to_display][self.train_groups.values == i].values.std()
            MAE = mean_absolute_error(self.y_train[self.label_to_display][self.train_groups.values == i].values,
                                      self.y_pred_train[self.label_to_display][self.train_groups.values == i].values)
            denominator = self.y_train[self.label_to_display][self.train_groups.values == i].values.mean()
            # if denominator < 1:
            #     denominator = 1
            rel_err = MAE * 100 / np.abs(denominator)

            results_df.loc[indx] = [ref, class_, mean, std, MAE, rel_err]

        df_length = results_df.shape[0]
        for indx, i in enumerate(np.unique(self.test_groups)):
            ref = self.y_test[self.label_to_display][self.test_groups.values == i].values.mean()
            class_ = 'Test'
            mean = self.y_pred_test[self.label_to_display][self.test_groups.values == i].values.mean()
            std = self.y_pred_test[self.label_to_display][self.test_groups.values == i].values.std()
            MAE = mean_absolute_error(self.y_test[self.label_to_display][self.test_groups.values == i].values,
                                      self.y_pred_test[self.label_to_display][self.test_groups.values == i].values)
            denominator = self.y_test[self.label_to_display][self.test_groups.values == i].values.mean()
            # if denominator < 1:
            #     denominator = 1
            rel_err = MAE * 100 / np.abs(denominator)

            results_df.loc[df_length + indx] = [ref, class_, mean, std, MAE, rel_err]

        print(f'Summary: {self.label_to_display} - PCs = {self.pls.best_estimator_.n_components}')
        print('------------------------')
        print(results_df)
        print('________________________________________________________________________')
        print(results_df.describe())

    def show_prediction_results(self):

        # y_predicted = PLS(self.df_x, self.df_y).validate_pls(self.pls, self.selected_labels)

        plt.figure(figsize=(13, 9))

        # Plot the data
        plt.plot(self.y_train.index, self.y_train[self.label_to_display],
                 'ok', markersize=3, label=f"Reference - {self.label_to_display}", alpha=0.5)

        plt.plot(self.y_train.index, self.y_pred_train[self.label_to_display],
                 'ob', markersize=3, label=f"Predicted Train - {self.label_to_display}", alpha=0.3)
        # plt.plot(self.y_train.index, self.y_pred_cv[self.label_to_display],
        #          'or', markersize=3, label=f"Predicted CV - {self.label_to_display}", alpha=0.3)
        if not self.y_test.empty:
            plt.plot(self.y_test.index, self.y_test[self.label_to_display],
                     'ok', markersize=3, alpha=0.5)
            plt.plot(self.y_test.index, self.y_pred_test[self.label_to_display],
                     'or', markersize=3, label=f"Predicted Test - {self.label_to_display}", alpha=0.3)

        # Add title and labels
        plt.title('Reference vs Predicted Data', fontsize=16)
        plt.xlabel('Sample Index', fontsize=14)
        plt.ylabel('Value', fontsize=14)

        # Add grid
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add legend
        plt.legend(fontsize=12)

        # Show plot with enhancements
        plt.show()

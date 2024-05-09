import _tkinter
import tkinter as tk
from tkinter import ttk, filedialog, END
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        self.select_all_button = ttk.Button(self.window, text="Select All", command=self.select_all)
        self.select_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def select_all(self):
        self.listbox.select_set(0, END)

    def select_columns(self):
        selected_indices = self.listbox.curselection()
        self.selected_columns = [self.columns[idx] for idx in selected_indices]

        self.window.destroy()


class PLS:
    def __init__(self, parent, df_X, df_y):
        self.components_range = None
        self.rmse_test = None
        self.rmse_train = None
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
        self.num_components = []
        self.y_pred_train = []
        self.y_pred_test = []
        self.r2_train = []
        self.r2_test = []
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
        self.train_test_ratio_entry.insert(tk.END, "0.2")

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

        self.components_range = range(1, 41)
        if max(self.components_range) > self.df_X.shape[1]:
            self.components_range = range(1, self.df_X.shape[1])

        # Iterate over different numbers of components
        for n_components in self.components_range:
            # Train the PLS model
            pls = PLSRegression(n_components=n_components)
            pls.fit(self.x_train, self.y_train)

            # Predict labels for the test data
            y_pred = pls.predict(self.x_test)

            # Calculate RMSE and append to the list
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            self.rmse_scores.append(rmse)

            # Calculate R2 score and append to the list
            r2 = r2_score(self.y_test, y_pred)
            self.r2_scores.append(r2)

        # Initialize PLS model with desired number of components
        self.pls = PLSRegression(n_components=self.num_components)

        # Fit the model
        self.pls.fit(self.x_train, self.y_train)

        # Predict on test set
        self.y_pred_train = self.pls.predict(self.x_train)
        self.y_pred_test = self.pls.predict(self.x_test)

        # Calculate R^2 score
        self.r2_train = r2_score(self.y_train, self.y_pred_train)
        self.r2_test = r2_score(self.y_test, self.y_pred_test)

        self.rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))

        self.window.destroy()


class MyChemometrix:
    def __init__(self, master):
        super().__init__()
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
        self.style.theme_use("alt")

        self.style.configure("TFrame", background="#F0F0F0")
        self.style.configure("TLabelFrame", background="#F0F0F0")
        self.style.configure("TButton", background="#4CAF50", foreground="black", font=("Arial", 10), padding=5)
        self.style.map("TButton", background=[("active", "#45A049")])

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
        self.top_left_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                         width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_left_canvas.grid(row=0, column=0, sticky="nsew")

        self.top_left_plot = FigureCanvasTkAgg(self.fig1, master=self.top_left_canvas)
        self.top_left_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        # Add line frame around the canvas widget
        self.top_left_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.top_left_plot.draw()

        self.top_right_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                          width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_right_canvas.grid(row=0, column=1, sticky="nsew")

        self.top_right_plot = FigureCanvasTkAgg(self.fig2, master=self.top_right_canvas)
        self.top_right_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.top_right_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.top_right_plot.draw()

        self.bottom_left_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                            width=graphs_frame_width // 2, height=screen_height // 2)
        self.bottom_left_canvas.grid(row=1, column=0, sticky="nsew")

        self.bottom_left_plot = FigureCanvasTkAgg(self.fig3, master=self.bottom_left_canvas)
        self.bottom_left_plot.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.bottom_left_plot.get_tk_widget().config(highlightthickness=1, highlightbackground="black")
        self.bottom_left_plot.draw()

        self.bottom_right_canvas = tk.Canvas(self.graphs_frame, bg="white",
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

        self.raw_data_tree = ttk.Treeview(self.raw_data_frame)
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

        self.labels_data_tree = ttk.Treeview(self.labels_data_frame)
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

        self.features_data_tree = ttk.Treeview(self.features_data_frame)
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

        self.display_raw_data_button = ttk.Button(self.data_buttons_frame, text="Display Data",
                                                  command=self.display_raw)
        self.display_raw_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.filter_data_button = ttk.Button(self.preprocessing_buttons_frame, text="Filter Data", state="disabled")
        self.filter_data_button.pack(side="left", fill="both", padx=5, pady=5)

        self.msc_button = ttk.Button(self.preprocessing_buttons_frame, text="MSC", state="disabled")
        self.msc_button.pack(side="left", fill="both", padx=5, pady=5)

        self.snv_button = ttk.Button(self.preprocessing_buttons_frame, text="SNV", state="disabled")
        self.snv_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pls_button = ttk.Button(self.regression_buttons_frame, text="PLS", state="disabled",
                                     command=self.open_pls_window)
        self.pls_button.pack(side="left", fill="both", padx=5, pady=5)

        self.optimize_button = ttk.Button(self.regression_buttons_frame, text="Optimize", state="disabled")
        self.optimize_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pca_button = ttk.Button(self.classifier_buttons_frame, text="PCA", state="disabled")
        self.pca_button.pack(side="left", fill="both", padx=5, pady=5)

        self.LDA_button = ttk.Button(self.classifier_buttons_frame, text="LDA", state="disabled")
        self.LDA_button.pack(side="left", fill="both", padx=5, pady=5)

        self.df_raw = None
        self.df_X = None
        self.df_y = None

    def import_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("csv and xlsx Files", "*.csv *.xlsx")])
        if file_path:
            self.display_data(file_path)

    def display_data(self, file_path):

        if file_path.endswith('.csv'):
            self.df_raw = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.df_raw = pd.read_excel(file_path)

        self.display_table(self.raw_data_tree, self.df_raw)

    def display_table(self, tree, df):
        try:
            tree.delete(*self.raw_data_tree.get_children())

        except _tkinter.TclError:
            pass

        tree['columns'] = df.iloc[:].columns.values.tolist()

        for col in tree['columns']:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", minwidth=100, width=100)

        for i, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        self.activate_buttons()

    def open_select_features_window(self):
        if self.df_raw is not None:
            select_features_window = SelectColumnsWindow(self.master, self.df_raw)
            self.master.wait_window(select_features_window.window)
            selected_columns = select_features_window.selected_columns
            self.df_X = self.df_raw[selected_columns]
            self.display_table(self.features_data_tree, self.df_X)
            self.activate_buttons()
        else:
            pass

    def open_select_labels_window(self):
        if self.df_raw is not None:
            select_labels_window = SelectColumnsWindow(self.master, self.df_raw)
            self.master.wait_window(select_labels_window.window)
            selected_columns = select_labels_window.selected_columns
            self.df_y = self.df_raw[selected_columns]
            self.display_table(self.labels_data_tree, self.df_y)
            self.activate_buttons()
        else:
            pass

    def activate_buttons(self):
        if self.df_raw is not None:
            self.select_y_button["state"] = "normal"
            self.select_x_button["state"] = "normal"
            if self.df_X is not None and self.df_y is not None:
                self.filter_data_button["state"] = "normal"
                self.msc_button["state"] = "normal"
                self.snv_button["state"] = "normal"
                self.pls_button["state"] = "normal"
                self.optimize_button["state"] = "normal"
                self.pca_button["state"] = "normal"
                self.LDA_button["state"] = "normal"

    def display_raw(self):
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
        normalized_dissimilarities_vector = (dissimilarities[1]-np.mean(dissimilarities[1]))/np.std(dissimilarities[1])

        ax3.scatter(self.df_X.index, normalized_dissimilarities_vector)
        ax3.set_title('Normalized Correlation Distance of X')
        ax3.set_xlabel('X rows [a.u]')
        ax3.set_ylabel('Amplitude [a.u]')
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))

        ax4.violinplot(self.df_y, showmeans=False, showmedians=True)
        ax4.boxplot(self.df_y, sym=".")

        ax4.set_xticks(range(1, len(self.df_y.columns) + 1))
        ax4.set_xticklabels(self.df_y.columns)

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()

    def open_pls_window(self):
        pls_window = PLS(self.master, self.df_X,
                         self.df_y)

        self.master.wait_window(pls_window.window)

        self.fig1.clear()

        ax1 = self.fig1.add_subplot(111)

        print(pls_window.rmse_train)
        print(pls_window.r2_train)

        # Creating the legend table
        legend_table = ax1.table(cellText=[[f'{pls_window.rmse_train:.6f}', f'{pls_window.r2_train:.6f}'],
                                           [f'{pls_window.rmse_test:.6f}', f'{pls_window.r2_test:.6f}']],
                                 colLabels=['RMSE', 'R-Square'],
                                 loc='upper left',
                                 cellLoc='center',
                                 cellColours=[['b', 'b'], ['r', 'r']])

        # Styling the legend table
        legend_table.auto_set_font_size(False)
        legend_table.set_fontsize(10)
        legend_table.scale(0.6, 1.1)  # Adjust the size of the legend table

        ax1.scatter(pls_window.y_train, pls_window.y_pred_train,
                    color='blue', s=1,
                    label=f'Trained: R2={pls_window.r2_train:.2f}\n'
                          f'         RMSE={pls_window.rmse_train:.2f}')
        ax1.scatter(pls_window.y_test, pls_window.y_pred_test,
                    color='red', s=1,
                    label=f'Tested: R2={pls_window.r2_test:.2f}')
        ax1.plot(pls_window.y_train, pls_window.y_train,
                 color='blue', linewidth=1, linestyle='--')
        ax1.plot(pls_window.y_test, pls_window.y_test,
                 color='red', linewidth=1, linestyle='--')
        ax1.set_title(f'Predicted vs True Results - Label={pls_window.selected_label} |'
                      f' LV={pls_window.num_components} |'
                      f' Test/Train={pls_window.train_test_ratio}')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        # ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))

        self.fig2.clear()

        # Find the index of the minimum RMSE score
        min_rmse_index = np.argmin(pls_window.rmse_scores)

        ax2 = self.fig2.add_subplot(111)
        ax2.plot(pls_window.components_range, pls_window.rmse_scores, marker='o')
        ax2.plot(pls_window.components_range[min_rmse_index], pls_window.rmse_scores[min_rmse_index],
                 'P', ms=10, mfc='red',
                 label=f'Optimized Number of Components={pls_window.components_range[min_rmse_index]}')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE vs Number of Components')
        ax2.legend(loc='best')
        ax2.grid(True)

        self.fig3.clear()

        x_scores = pls_window.pls.x_scores_
        y_scores = pls_window.pls.y_scores_

        ax3 = self.fig3.add_subplot(111)
        ax3.plot(x_scores[:, 0], x_scores[:, 1], 'ob', label="X scores")
        ax3.plot(y_scores[:, 0], y_scores[:, 1], 'or', label='y scores')

        ax3.set_xlabel('Component 1')
        ax3.set_ylabel('Component 2')
        ax3.set_title('Scores Plot')
        ax3.grid(True)

        self.fig4.clear()

        x_loadings = pls_window.pls.x_loadings_
        y_loadings = pls_window.pls.y_loadings_
        ax4 = self.fig4.add_subplot(111)
        ax4.plot(x_loadings[:, 0], x_loadings[:, 1], 'ob')
        ax4.plot(y_loadings[:, 0], y_loadings[:, 1], 'or')
        ax4.set_xlabel('Component 1')
        ax4.set_ylabel('Component 2')
        ax4.set_title('Loadings Plot')

        self.top_left_plot.draw()
        self.top_right_plot.draw()
        self.bottom_left_plot.draw()
        self.bottom_right_plot.draw()


def main():
    root = tk.Tk()
    app = MyChemometrix(root)
    root.mainloop()


if __name__ == "__main__":
    main()

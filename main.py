import _tkinter
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SelectColumnsWindow:
    def __init__(self, parent, df_raw):
        self.parent = parent
        self.columns = df_raw.columns
        self.selected_columns = []

        self.window = tk.Toplevel(parent)
        self.window.title("Select Columns")
        self.window.geometry("200x400")

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

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def select_columns(self):
        selected_indices = self.listbox.curselection()
        self.selected_columns = [self.columns[idx] for idx in selected_indices]

        self.window.destroy()


class SelectPlsOptions:
    def __init__(self, parent, df_X, df_y):
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

        self.window = tk.Toplevel(parent)
        self.window.title("PLS Options")
        self.window.geometry("320x200")

        # Label and entry for number of components
        self.num_components_label = ttk.Label(self.window, text="Number of Components:")
        self.num_components_label.grid(row=0, column=0, padx=5, pady=5)

        self.num_components_entry = ttk.Entry(self.window, width=10)
        self.num_components_entry.grid(row=0, column=1, padx=5, pady=5)
        self.num_components_entry.insert(tk.END, "2")

        # ComboBox for selecting label
        self.select_label_label = ttk.Label(self.window, text="Select Label:")
        self.select_label_label.grid(row=1, column=0, padx=5, pady=5)

        self.select_label_combobox = ttk.Combobox(self.window, values=self.df_y.columns.tolist())
        self.select_label_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.select_label_combobox.current(0)

        # Apply and Cancel buttons
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_pls)
        self.apply_button.grid(row=2, column=0, padx=5, pady=5)

        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.window.destroy)
        self.cancel_button.grid(row=2, column=1, padx=5, pady=5)

    def apply_pls(self):

        # Split data into train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df_X, self.df_y,
                                                                                test_size=0.2)
        self.num_components = int(self.num_components_entry.get())
        selected_label = self.select_label_combobox.get()

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
        self.paned_window = ttk.Panedwindow(master, orient="horizontal")
        self.paned_window.pack(expand=True, fill="both")

        self.data_frame = ttk.LabelFrame(self.paned_window, text="Data", width=data_frame_width)
        self.paned_window.add(self.data_frame)

        self.graphs_frame = ttk.LabelFrame(self.paned_window, text="Graphics", width=graphs_frame_width)
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
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
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

    def open_pls_window(self):
        select_pls_options_window = SelectPlsOptions(self.master, self.df_X,
                                                     self.df_y)

        self.master.wait_window(select_pls_options_window.window)

        ax1 = self.fig1.add_subplot(111)
        ax1.scatter(select_pls_options_window.y_train, select_pls_options_window.y_pred_train,
                    color='blue', s=1,
                    label=f'Trained: R2 Score: {select_pls_options_window.r2_train:.2f}')
        ax1.scatter(select_pls_options_window.y_test, select_pls_options_window.y_pred_test,
                    color='red', s=1,
                    label=f'Tested: R2 Score: {select_pls_options_window.r2_test:.2f}')
        ax1.plot(select_pls_options_window.y_train, select_pls_options_window.y_train,
                 color='blue', linewidth=1, linestyle='--')
        ax1.plot(select_pls_options_window.y_test, select_pls_options_window.y_test,
                 color='red', linewidth=1, linestyle='--')
        ax1.set_title('Predicted vs True Results')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        self.top_left_plot.draw()


def main():
    root = tk.Tk()
    app = MyChemometrix(root)
    root.mainloop()


if __name__ == "__main__":
    main()

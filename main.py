import _tkinter
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd


class SelectFeaturesWindow:
    def __init__(self, parent, df_raw):
        self.parent = parent
        self.columns = df_raw.columns
        self.selected_columns = []

        self.window = tk.Toplevel(parent)
        self.window.title("Select Features")
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


class MyChemometrix:
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.master.title("MyChemometrix")

        # Get screen size
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

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

        # Canvas areas in the graphs_frame
        self.top_left_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                         width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_left_canvas.grid(row=0, column=0, sticky="nsew")

        self.top_right_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                          width=graphs_frame_width // 2, height=screen_height // 2)
        self.top_right_canvas.grid(row=0, column=1, sticky="nsew")

        self.bottom_left_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                            width=graphs_frame_width // 2, height=screen_height // 2)
        self.bottom_left_canvas.grid(row=1, column=0, sticky="nsew")

        self.bottom_right_canvas = tk.Canvas(self.graphs_frame, bg="white",
                                             width=graphs_frame_width // 2, height=screen_height // 2)
        self.bottom_right_canvas.grid(row=1, column=1, sticky="nsew")

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

        self.label_select_button = ttk.Button(self.regression_buttons_frame, text="Select Label", state="disabled")
        self.label_select_button.pack(side="left", fill="both", padx=5, pady=5)

        self.pls_button = ttk.Button(self.regression_buttons_frame, text="PLS", state="disabled")
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
            select_features_window = SelectFeaturesWindow(self.master, self.df_raw)
            self.master.wait_window(select_features_window.window)
            selected_columns = select_features_window.selected_columns
            self.df_X = self.df_raw[selected_columns]
            self.display_table(self.features_data_tree, self.df_X)
        else:
            pass

    def open_select_labels_window(self):
        if self.df_raw is not None:
            select_features_window = SelectFeaturesWindow(self.master, self.df_raw)
            self.master.wait_window(select_features_window.window)
            selected_columns = select_features_window.selected_columns
            self.df_y = self.df_raw[selected_columns]
            self.display_table(self.labels_data_tree, self.df_y)
        else:
            pass

    def activate_buttons(self):
        if self.df_raw is not None:
            self.select_y_button["state"] = "normal"
            self.select_x_button["state"] = "normal"
            # self.filter_data_button["state"] = "normal"
            # self.msc_button["state"] = "normal"
            # self.snv_button["state"] = "normal"
            # self.label_select_button["state"] = "normal"
            # self.pls_button["state"] = "normal"
            # self.optimize_button["state"] = "normal"
            # self.pca_button["state"] = "normal"
            # self.LDA_button["state"] = "normal"


def main():
    root = tk.Tk()
    app = MyChemometrix(root)
    root.mainloop()


if __name__ == "__main__":
    main()

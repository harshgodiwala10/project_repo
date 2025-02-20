import os
import pandas as pd
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk as tk_tt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
from scipy.stats import gmean

class CSVProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Processor")
        self.root.geometry("1280x900")

        # Using ttkbootstrap theme
        self.style = ttk.Style()
        self.style.theme_use("cosmo")

        # Store Data
        self.df_original = None
        self.df_processed = None
        self.selected_columns = []
        self.label_column = None

        # Create Pages
        self.page_upload = ctk.CTkFrame(root)
        self.page_data_view = ctk.CTkFrame(root)
        self.page_column_select = ctk.CTkFrame(root)
        self.page_results = ctk.CTkFrame(root)

        # Create UI Elements
        self.create_upload_page()
        self.create_data_view_page()
        self.create_column_selection_page()
        self.create_results_page()

        # Show Upload Page First
        self.show_page(self.page_upload)

    def create_upload_page(self):
        """ Page 1: Upload CSV """
        ctk.CTkLabel(self.page_upload, text="Upload CSV File", font=("Helvetica", 28, "bold")).pack(pady=20)
        ctk.CTkButton(self.page_upload, text="Select File", command=self.load_csv, width=250, fg_color="#4CAF50").pack(pady=10)
        self.page_upload.pack(fill="both", expand=True)

    def create_data_view_page(self):
        """ Page 2: Show Data """
        ctk.CTkLabel(self.page_data_view, text="Uploaded CSV Data", font=("Helvetica", 28, "bold")).pack(pady=20)

        # Data Frame Display
        self.frame_table_original = ctk.CTkFrame(self.page_data_view)
        self.frame_table_original.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkButton(self.page_data_view, text="Next: Select Label and Features", command=self.show_page_column_select, width=250, fg_color="#2196F3").pack(pady=10)

    def create_column_selection_page(self):
        """ Page 3: Select Label and Features """
        ctk.CTkLabel(self.page_column_select, text="Select Label and Features", font=("Helvetica", 28, "bold")).pack(pady=20)

        self.label_var = ttk.StringVar()
        self.column_vars = []

        self.label_frame = ctk.CTkFrame(self.page_column_select)
        self.label_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.column_frame = ctk.CTkFrame(self.page_column_select)
        self.column_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.process_button = ctk.CTkButton(self.page_column_select, text="Process Data", command=self.process_data, width=250, fg_color="#2196F3")
        self.process_button.pack(pady=10)

    def create_results_page(self):
        """ Final Page: Show Processed and Original Data """
        self.nav_frame = ctk.CTkFrame(self.page_results)
        self.nav_frame.pack(fill="x", padx=20, pady=10)

        self.btn_original = ctk.CTkButton(self.nav_frame, text="View Original Data", command=lambda: self.show_table(self.df_original, "Original Data"), width=200, state="disabled")
        self.btn_original.pack(side="left", padx=10)

        self.btn_processed = ctk.CTkButton(self.nav_frame, text="View Processed Data", command=lambda: self.show_table(self.df_processed, "Processed Data"), width=200, state="disabled")
        self.btn_processed.pack(side="left", padx=10)

        self.result_table_frame = ctk.CTkFrame(self.page_results)
        self.result_table_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def show_page(self, page):
        """ Hide all pages and show the selected one """
        for p in [self.page_upload, self.page_data_view, self.page_column_select, self.page_results]:
            p.pack_forget()
        page.pack(fill="both", expand=True)

    def load_csv(self):
        """ Load CSV file and show data """
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            self.display_table(self.df_original, self.frame_table_original, "Original Data")
            self.show_page(self.page_data_view)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def show_page_column_select(self):
        """ Populate label and feature selection checkboxes """
        for widget in self.label_frame.winfo_children():
            widget.destroy()

        for widget in self.column_frame.winfo_children():
            widget.destroy()

        self.column_vars = []

        # Create Label Selection Dropdown
        ctk.CTkLabel(self.label_frame, text="Select Label Column", font=("Arial", 16, "bold")).pack(pady=10)
        label_options = list(self.df_original.columns)
        label_menu = ttk.Combobox(self.label_frame, values=label_options, textvariable=self.label_var, state="readonly")
        label_menu.pack(pady=10)
        label_menu.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Create Feature Selection Checkboxes
        self.feature_vars = []
        self.update_feature_selection()

        self.show_page(self.page_column_select)

    def update_feature_selection(self, event=None):
        """ Update the feature selection list based on the selected label """
        label_column = self.label_var.get()

        # Reset feature selection
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        self.feature_vars = []
        remaining_columns = [col for col in self.df_original.columns if col != label_column]

        for col in remaining_columns:
            var = ttk.IntVar()
            ttk.Checkbutton(self.column_frame, text=col, variable=var, bootstyle="primary").pack(anchor="w", padx=20)
            self.feature_vars.append((col, var))

    def process_data(self):
        """ Process selected label and features """
        label_column = self.label_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get() == 1]

        if not label_column or not selected_features:
            messagebox.showerror("Error", "Please select a label column and at least one feature for processing!")
            return

        self.df_processed = self.preprocess_data(self.df_original[selected_features + [label_column]].copy())
        if self.df_processed is not None:
            self.btn_original.configure(state="normal")
            self.btn_processed.configure(state="normal")
            self.show_page(self.page_results)

    def preprocess_data(self, data):
        """ Preprocessing Steps """
        try:
            # Handle missing values
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing_values = data[col].dropna()
                positive_values = non_missing_values[non_missing_values > 0]
                if len(positive_values) > 0:
                    geometric_mean = gmean(positive_values)
                    data.loc[data[col].isna(), col] = geometric_mean

            # Standardization
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            joblib.dump(scaler, "scaler.pkl")

            # Encoding categorical data
            label_encoders = {}
            for col in data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le
            joblib.dump(label_encoders, "encoder.pkl")

            return data
        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {e}")
            return None

    def display_table(self, df, frame, title):
        """ Display Data Table """
        for widget in frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(frame, text=title, font=("Arial", 16, "bold")).pack()
        tree = tk_tt.Treeview(frame, columns=df.columns.tolist(), show="headings", height=15)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        for col in df.columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)

        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def show_table(self, df, title):
        """ Show table inside results page """
        self.display_table(df, self.result_table_frame, title)

if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVProcessorApp(root)
    root.mainloop()


# import os
# import pandas as pd
# import ttkbootstrap as ttk
# import customtkinter as ctk
# from tkinter import messagebox, filedialog, ttk as tk_tt
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import joblib
# import numpy as np
# from scipy.stats import gmean
#
# class CSVProcessorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("CSV Data Processor")
#         self.root.geometry("1280x900")
#
#         # Using ttkbootstrap theme
#         self.style = ttk.Style()
#         self.style.theme_use("cosmo")
#
#         # Set Background Color
#         self.root.configure(bg="#e0f7fa")
#
#         # Store Data
#         self.df_original = None
#         self.df_processed = None
#         self.selected_columns = []
#         self.label_column = None
#
#         # Create Pages
#         self.page_upload = ctk.CTkFrame(root, fg_color="#e0f7fa")
#         self.page_data_view = ctk.CTkFrame(root)
#         self.page_column_select = ctk.CTkFrame(root)
#         self.page_results = ctk.CTkFrame(root)
#
#         # Create UI Elements
#         self.create_upload_page()
#         self.create_data_view_page()
#         self.create_column_selection_page()
#         self.create_results_page()
#
#         # Show Upload Page First
#         self.show_page(self.page_upload)
#
#     def create_upload_page(self):
#         """ Page 1: Upload CSV """
#         # Center the upload section
#         upload_frame = ctk.CTkFrame(self.page_upload, fg_color="#ffffff", corner_radius=15)
#         upload_frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=200)
#
#         # Title and Button in Center
#         ctk.CTkLabel(upload_frame, text="Upload CSV File", font=("Helvetica", 28, "bold"), fg_color="#ffffff").pack(pady=20)
#         ctk.CTkButton(upload_frame, text="Select File", command=self.load_csv, width=250, fg_color="#4CAF50").pack(pady=10)
#
#         self.page_upload.pack(fill="both", expand=True)
#
#     def create_data_view_page(self):
#         """ Page 2: Show Data """
#         ctk.CTkLabel(self.page_data_view, text="Uploaded CSV Data", font=("Helvetica", 28, "bold")).pack(pady=20)
#
#         # Data Frame Display
#         self.frame_table_original = ctk.CTkFrame(self.page_data_view)
#         self.frame_table_original.pack(fill="both", expand=True, padx=20, pady=10)
#
#         ctk.CTkButton(self.page_data_view, text="Next: Select Label and Features", command=self.show_page_column_select, width=250, fg_color="#2196F3").pack(pady=10)
#
#     def show_page(self, page):
#         """ Hide all pages and show the selected one """
#         for p in [self.page_upload, self.page_data_view, self.page_column_select, self.page_results]:
#             p.pack_forget()
#         page.pack(fill="both", expand=True)
#
#     def load_csv(self):
#         """ Load CSV file and show data """
#         file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#         if not file_path:
#             return
#
#         try:
#             self.df_original = pd.read_csv(file_path)
#             messagebox.showinfo("Success", "CSV file loaded successfully!")
#             self.show_page(self.page_data_view)
#         except Exception as e:
#             messagebox.showerror("Error", f"Could not load file: {e}")
#
#     def create_column_selection_page(self):
#         pass
#
# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = CSVProcessorApp(root)
#     root.mainloop()

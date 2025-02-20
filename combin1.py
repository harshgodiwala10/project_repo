import os
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
from scipy.stats import gmean

class CSVPreprocessor:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Preprocessing Tool")
        self.root.geometry("1200x800")
        ctk.set_default_color_theme("dark-blue")  # Improved color theme

        # Title
        ctk.CTkLabel(root, text="CSV Data Preprocessing Tool", font=("Helvetica", 30, "bold"), text_color="#000000").pack(pady=20)

        # Buttons Frame
        self.frame_buttons = ctk.CTkFrame(root, fg_color="#222222")  # Darker background for contrast
        self.frame_buttons.pack(fill='x', padx=20, pady=10)

        ctk.CTkButton(self.frame_buttons, text="Upload CSV", command=self.load_csv, width=200, fg_color="#4CAF50").pack(side="left", padx=15)
        ctk.CTkButton(self.frame_buttons, text="Preprocess Data", command=self.process_data, width=200, fg_color="#2196F3").pack(side="left", padx=15)
        ctk.CTkButton(self.frame_buttons, text="Exit", command=root.quit, width=200, fg_color="#F44336").pack(side="left", padx=15)

        # Table Frames
        self.frame_data = ctk.CTkFrame(root, fg_color="#333333")
        self.frame_data.pack(fill='both', expand=True, padx=20, pady=10)

        # Checkboxes for column selection
        self.frame_columns = ctk.CTkFrame(root, fg_color="#444444")
        self.frame_columns.pack(fill='x', padx=20, pady=10)
        ctk.CTkLabel(self.frame_columns, text="Select Columns for Preprocessing:", font=("Arial", 16), text_color="#ffffff").pack()
        self.column_vars = {}

        self.df_original = None
        self.df_processed = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            self.display_table(self.df_original, "Original Data")
            self.create_column_checkboxes()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def create_column_checkboxes(self):
        for widget in self.frame_columns.winfo_children()[1:]:
            widget.destroy()
        
        self.column_vars.clear()
        for col in self.df_original.columns:
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(self.frame_columns, text=col, variable=var, text_color="#ffffff")
            chk.pack(anchor="w", padx=10)
            self.column_vars[col] = var

    def preprocess_data(self, data):
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            selected_columns = list(data.columns)  # Use all columns if none selected
        
        data = data[selected_columns].copy()
        
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            non_missing_values = data[col].dropna()
            positive_values = non_missing_values[non_missing_values > 0]
            if len(positive_values) > 0:
                geometric_mean = gmean(positive_values)
                data.loc[data[col].isna(), col] = geometric_mean

        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        joblib.dump(scaler, "scaler.pkl")
        
        label_encoders = {}
        for col in data.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        joblib.dump(label_encoders, "encoder.pkl")
        
        pca = PCA(n_components=min(len(numeric_cols), 5))
        if len(numeric_cols) > 1:
            pca_data = pca.fit_transform(data[numeric_cols])
            pca_columns = [f"PCA_{i+1}" for i in range(pca_data.shape[1])]
            pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
            data.drop(columns=numeric_cols, inplace=True)
            data = pd.concat([data, pca_df], axis=1)
            joblib.dump(pca, "PCA.pkl")
        
        return data

    def process_data(self):
        if self.df_original is None:
            messagebox.showerror("Error", "No CSV file loaded!")
            return

        self.df_processed = self.preprocess_data(self.df_original.copy())
        if self.df_processed is not None:
            self.display_table(self.df_processed, "Preprocessed Data")

    def display_table(self, df, title):
        for widget in self.frame_data.winfo_children():
            widget.destroy()

        ctk.CTkLabel(self.frame_data, text=title, font=("Arial", 16, "bold"), text_color="#ffffff").pack()
        frame = ctk.CTkFrame(self.frame_data)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = ctk.CTkCanvas(frame)
        scrollbar = ctk.CTkScrollbar(frame, orientation="vertical", command=canvas.yview)
        scroll_frame = ctk.CTkFrame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        columns = df.columns.tolist()
        tree = ttk.Treeview(scroll_frame, columns=columns, show="headings", height=15)
        tree.pack()

        for col in columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)

        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVPreprocessor(root)
    root.mainloop()

import os
import pandas as pd
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, StringVar
import tkinter.ttk as tk_tt
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

        # Colors & Font Settings
        self.bg_color = "#1e1e2f"         # Dark background
        self.text_color = "#ffffff"       # White text
        self.accent_color = "#6c5ce7"     # Accent (purple)
        self.secondary_color = "#2d2d44"  # Secondary frames
        self.font_heading = ("Times New Roman", 32, "bold")
        self.font_subheading = ("Times New Roman", 24)
        self.font_normal = ("Times New Roman", 20)

        # Create Pages
        self.page_upload = ctk.CTkFrame(root, fg_color=self.bg_color)
        self.page_data_view = ctk.CTkFrame(root, fg_color=self.bg_color)
        self.page_column_select = ctk.CTkFrame(root, fg_color=self.bg_color)
        self.page_results = ctk.CTkFrame(root, fg_color=self.bg_color)

        # Create UI Elements on each page
        self.create_upload_page()
        self.create_data_view_page()
        self.create_column_selection_page()
        self.create_results_page()

        # Show Upload Page First
        self.show_page(self.page_upload)

    def create_upload_page(self):
        """Page 1: Upload CSV"""
        self.page_upload.configure(fg_color=self.bg_color)
        # Center Frame for upload elements
        center_frame = ctk.CTkFrame(self.page_upload, fg_color="transparent")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Title
        ctk.CTkLabel(center_frame,
                     text="CSV Data Processor",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)
        # Subtitle
        ctk.CTkLabel(center_frame,
                     text="Upload your CSV file to begin processing",
                     font=self.font_subheading,
                     text_color="#a1a1a1").pack(pady=10)
        # Select File Button
        ctk.CTkButton(center_frame,
                      text="Select File",
                      command=self.load_csv,
                      font=self.font_normal,
                      width=300,
                      height=50,
                      fg_color=self.accent_color,
                      hover_color="#5a4dbf").pack(pady=20)
        # Version info (optional)
        ctk.CTkLabel(center_frame,
                     text="Version 1.0",
                     font=("Times New Roman", 14),
                     text_color="#a1a1a1").pack(pady=5)
        self.page_upload.pack(fill="both", expand=True)

    def create_data_view_page(self):
        """Page 2: Display Uploaded CSV Data"""
        ctk.CTkLabel(self.page_data_view,
                     text="Uploaded CSV Data",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)
        # Frame for displaying table
        self.frame_table_original = ctk.CTkFrame(self.page_data_view, fg_color=self.secondary_color)
        self.frame_table_original.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkButton(self.page_data_view,
                      text="Select Label and Features",
                      command=self.show_page_column_select,
                      width=250,
                      font=self.font_normal,
                      fg_color=self.accent_color).pack(pady=10)

    def create_column_selection_page(self):
        """Page 3: Select Label and Features"""
        ctk.CTkLabel(self.page_column_select,
                     text="Select Label",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)
        self.label_var = StringVar()  # Use StringVar from tkinter
        self.column_vars = []

        # Frame for label selection
        self.label_frame = ctk.CTkFrame(self.page_column_select, fg_color=self.secondary_color)
        self.label_frame.pack(fill="x", expand=False, padx=20, pady=10)
        
        # Use CTkComboBox for better looks and bigger text
        # (It will be created in show_page_column_select)
        
        ctk.CTkLabel(self.page_column_select,
                     text="Select Features",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)
        
        # Frame for feature selection with scrollable frame
        # We use a CTkScrollableFrame to handle many features
        self.feature_scrollable_frame = ctk.CTkScrollableFrame(self.page_column_select,
                                                                fg_color=self.secondary_color,
                                                                width=600,
                                                                height=300)
        self.feature_scrollable_frame.pack(fill="x", expand=False, padx=20, pady=10)

        # Buttons for Select All / Unselect All
        btn_frame = ctk.CTkFrame(self.page_column_select, fg_color="transparent")
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame,
                      text="Select All",
                      command=self.select_all_features,
                      width=150,
                      font=self.font_normal,
                      fg_color=self.accent_color).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame,
                      text="Unselect All",
                      command=self.unselect_all_features,
                      width=150,
                      font=self.font_normal,
                      fg_color=self.accent_color).pack(side="left", padx=10)

        # Process Button (placed at the bottom so it stays visible)
        self.process_button = ctk.CTkButton(self.page_column_select,
                                            text="Process Data",
                                            command=self.process_data,
                                            width=250,
                                            font=self.font_normal,
                                            fg_color=self.accent_color)
        self.process_button.pack(pady=20)

    def create_results_page(self):
        """Page 4: Show Processed and Original Data"""
        self.nav_frame = ctk.CTkFrame(self.page_results, fg_color="transparent")
        self.nav_frame.pack(fill="x", padx=20, pady=10)
        self.btn_original = ctk.CTkButton(self.nav_frame,
                                          text="View Original Data",
                                          command=lambda: self.show_table(self.df_original, "Original Data"),
                                          width=200,
                                          font=self.font_normal,
                                          state="disabled")
        self.btn_original.pack(side="left", padx=10)
        self.btn_processed = ctk.CTkButton(self.nav_frame,
                                           text="View Processed Data",
                                           command=lambda: self.show_table(self.df_processed, "Processed Data"),
                                           width=200,
                                           font=self.font_normal,
                                           state="disabled")
        self.btn_processed.pack(side="left", padx=10)
        self.result_table_frame = ctk.CTkFrame(self.page_results, fg_color=self.secondary_color)
        self.result_table_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def show_page(self, page):
        """Hide all pages and show the selected one."""
        for p in [self.page_upload, self.page_data_view, self.page_column_select, self.page_results]:
            p.pack_forget()
        page.pack(fill="both", expand=True)

    def load_csv(self):
        """Load CSV file and display data."""
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
        """Populate label and feature selection widgets."""
        # Clear previous widgets in label and feature frames
        for widget in self.label_frame.winfo_children():
            widget.destroy()
        for widget in self.feature_scrollable_frame.winfo_children():
            widget.destroy()

        # Recreate label selection dropdown using CTkComboBox
        label_options = list(self.df_original.columns)
        self.ctk_label_combo = ctk.CTkComboBox(self.label_frame,
                                               values=label_options,
                                               variable=self.label_var,
                                               font=("Times New Roman", 18),
                                               dropdown_font=("Times New Roman", 18),
                                               width=400,
                                               height=40)
        self.ctk_label_combo.pack(pady=10)
        self.ctk_label_combo.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Create feature selection checkboxes in the scrollable frame
        self.update_feature_selection()

        self.show_page(self.page_column_select)

    def update_feature_selection(self, event=None):
        """Update the feature selection checkboxes based on the selected label."""
        label_column = self.label_var.get()
        # Clear previous feature selection widgets from scrollable frame
        for widget in self.feature_scrollable_frame.winfo_children():
            widget.destroy()
        self.feature_vars = []
        remaining_columns = [col for col in self.df_original.columns if col != label_column]
        for col in remaining_columns:
            var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(self.feature_scrollable_frame,
                                  text=col,
                                  variable=var,
                                  font=self.font_normal,
                                  text_color=self.text_color,
                                  checkbox_height=25,
                                  checkbox_width=25,
                                  hover_color=self.accent_color)
            chk.pack(anchor="w", padx=20, pady=5)
            self.feature_vars.append((col, var))

    def select_all_features(self):
        """Select all feature checkboxes."""
        for _, var in self.feature_vars:
            var.set(True)

    def unselect_all_features(self):
        """Deselect all feature checkboxes."""
        for _, var in self.feature_vars:
            var.set(False)

    def process_data(self):
        """Process selected label and features."""
        label_column = self.label_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get()]
        if not label_column or not selected_features:
            messagebox.showerror("Error", "Please select a label column and at least one feature for processing!")
            return
        self.df_processed = self.preprocess_data(self.df_original[selected_features + [label_column]].copy())
        if self.df_processed is not None:
            self.btn_original.configure(state="normal")
            self.btn_processed.configure(state="normal")
            self.show_page(self.page_results)

    def preprocess_data(self, data):
        """Preprocessing Steps."""
        try:
            # Handle missing numeric values using geometric mean for positive values
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing = data[col].dropna()
                positive = non_missing[non_missing > 0]
                if len(positive) > 0:
                    geometric_mean = gmean(positive)
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
        """Display Data Table."""
        for widget in frame.winfo_children():
            widget.destroy()
        ctk.CTkLabel(frame,
                     text=title,
                     font=("Times New Roman", 20, "bold"),
                     text_color=self.text_color).pack(pady=10)
        tree = tk_tt.Treeview(frame,
                              columns=df.columns.tolist(),
                              show="headings",
                              height=15)
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        for col in df.columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)
        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def show_table(self, df, title):
        """Show table inside results page."""
        self.display_table(df, self.result_table_frame, title)


if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVProcessorApp(root)
    root.mainloop()

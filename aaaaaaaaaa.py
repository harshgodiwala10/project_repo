import os
import pandas as pd
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, StringVar
import tkinter.ttk as tk_tt

from fontTools.varLib import models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
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
        self.bg_color = "#1e1e2f"  # Dark background
        self.text_color = "#ffffff"  # White text
        self.accent_color = "#6c5ce7"  # Accent (purple)
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
        center_frame = ctk.CTkFrame(self.page_upload, fg_color="transparent")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(center_frame,
                     text="CSV Data Processor",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)

        ctk.CTkButton(center_frame,
                      text="Select File",
                      command=self.load_csv,
                      font=self.font_normal,
                      width=300,
                      height=50,
                      fg_color=self.accent_color,
                      hover_color="#5a4dbf").pack(pady=20)

    def create_data_view_page(self):
        """Page 2: Display Uploaded CSV Data"""
        ctk.CTkLabel(self.page_data_view,
                     text="Uploaded CSV Data",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)

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

        self.label_var = StringVar()

        # Frame for label selection
        self.label_frame = ctk.CTkFrame(self.page_column_select, fg_color=self.secondary_color)
        self.label_frame.pack(fill="x", expand=False, padx=20, pady=10)

        ctk.CTkLabel(self.page_column_select,
                     text="Select Features",
                     font=self.font_heading,
                     text_color=self.text_color).pack(pady=20)

        # Scrollable frame for feature selection checkboxes
        self.feature_scrollable_frame = ctk.CTkScrollableFrame(self.page_column_select,
                                                               fg_color=self.secondary_color,
                                                               width=600,
                                                               height=300)
        self.feature_scrollable_frame.pack(fill="x", expand=False, padx=20, pady=10)

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

    def create_results_page(self):
        """Page 4: Show Processed and Original Data"""

    def show_page(self, page):
        """Hide all pages and show the selected one."""
        for p in [self.page_upload, self.page_data_view, self.page_column_select]:
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
        for widget in self.label_frame.winfo_children():
            widget.destroy()
        for widget in self.feature_scrollable_frame.winfo_children():
            widget.destroy()

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
        remaining_columns = [col for col in label_options if col != label_column]
        for col in remaining_columns:
            var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(self.feature_scrollable_frame,
                                  text=col,
                                  variable=var)
            chk.pack(anchor="w")
            self.feature_vars.append((col, var))

    def update_feature_selection(self, event=None):
        """Update the feature selection checkboxes based on the selected label."""
        label_column = self.label_var.get()

        for widget in self.feature_scrollable_frame.winfo_children():
            widget.destroy()

        remaining_columns = [col for col in self.df_original.columns if col != label_column]
        for col in remaining_columns:
            var = ctk.BooleanVar(value=False)
            chk = ctk.CTkCheckBox(self.feature_scrollable_frame,
                                  text=col,
                                  variable=var)
            chk.pack(anchor="w")

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

        # Preprocess data and store processed data.
        try:
            df_selected = pd.concat([self.df_original[selected_features],
                                     self.df_original[label_column]], axis=1)

            # Preprocess the data using the defined method.
            processed_data = preprocess_data(df_selected.copy())

            if processed_data is not None:
                # Store processed data in class variable.
                self.df_processed = processed_data

                # Enable buttons to view original and processed data.
                messagebox.showinfo("Success", "Data processed successfully!")
                print("Processed Data: \n", processed_data.head())

                # Proceed to classification.
                results_summary = classify_models(processed_data, label_column)

                # Display results summary (you can customize this part).
                messagebox.showinfo("Classification Results", results_summary)

                # Show results page.
                # You can create a method to display these results nicely.

        except Exception as e:
            messagebox.showerror("Error during processing", str(e))


def preprocess_data(data):
    """Preprocessing Steps."""
    try:

        # Handle missing numeric values using geometric mean for positive values.
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            non_missing_values = data[col].dropna()
            positive_values = non_missing_values[non_missing_values > 0]
            if len(positive_values) > 0:
                geometric_mean_value = gmean(positive_values)
                data.loc[data[col].isna(), col] = geometric_mean_value

        # Standardization.
        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        joblib.dump(scaler, "scaler.pkl")

        # Encoding categorical data.
        label_encoders_dicts = {}
        for col in data.select_dtypes(include=["object"]).columns.tolist():
            le_encoder_instance = LabelEncoder()
            data[col] = le_encoder_instance.fit_transform(data[col])
            label_encoders_dicts[col] = le_encoder_instance

        joblib.dump(label_encoders_dicts, "encoder.pkl")

        return data

    except Exception as e:
        messagebox.showerror("Error during preprocessing: ", str(e))
        return None


def classify_models(dataframe, label_col):
    """Classify using various models and return summary of results."""

    X = dataframe.drop(columns=[label_col])
    y = dataframe[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Machine': SVC()
    }


results_summary = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = f1_score(y_test, predictions, multi_class='weighted')
    report_str = f"{model_name}:\nAccuracy: {accuracy:.4f}\n{classification_report(y_test, predictions)}"

    results_summary.append(report_str)

    # return "\n\n".join(results_summary)

if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVProcessorApp(root)
    root.mainloop()

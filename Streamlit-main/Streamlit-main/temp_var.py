from statistics import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk as tk_tt
current_plot = 0
from pandas.core.common import random_state
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report, r2_score, explained_variance_score, \
    max_error
import joblib
import final_classification_pipe as fcp
# from eval_2 import final_regression_pipe as frp
import numpy as np
from scipy.stats import gmean

# Configure CustomTkinter appearance
ctk.set_appearance_mode("light")  # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Other available themes: "green", "dark-blue"

class CSVProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Processor - Classification")
        self.root.geometry("1280x900")
        self.root.minsize(1000, 700)

        # Using ttkbootstrap theme
        self.style = ttk.Style()
        self.style.theme_use("cosmo")

        # Data storage
        self.df_original = None
        self.df_processed = None
        self.global_model_results = None  # To store model results

        # Create pages
        self.create_pages()
        # Create navigation bar
        self.create_nav_bar()

        # Show Upload Page first
        self.show_page(self.page_upload)

    def create_nav_bar(self):
        """Top navigation bar for easy page switching."""
        self.nav_bar = ctk.CTkFrame(self.root, height=50)
        self.nav_bar.pack(side="top", fill="x", padx=10, pady=(10, 0))

        btn_style = {"width": 180, "corner_radius": 6, "fg_color": "#3B8ED0"}

        ctk.CTkButton(
            self.nav_bar, text="Upload CSV", command=lambda: self.show_page(self.page_upload), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="View Data", command=lambda: self.show_page(self.page_data_view), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Select Columns", command=lambda: self.show_page(self.page_column_select), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Results", command=lambda: self.show_page(self.page_results), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Run Classification", command=lambda: self.show_page(self.page_model_select), **btn_style
        ).pack(side="left", padx=5)

    def create_pages(self):
        """Initialize all pages."""
        self.page_upload = ctk.CTkFrame(self.root)
        self.page_data_view = ctk.CTkFrame(self.root)
        self.page_column_select = ctk.CTkFrame(self.root)
        self.page_results = ctk.CTkFrame(self.root)
        self.page_model_select = ctk.CTkFrame(self.root)
        self.page_model_results = ctk.CTkFrame(self.root)

        self.create_upload_page()
        self.create_data_view_page()
        self.create_column_selection_page()
        self.create_results_page()
        self.create_model_selection_page()
        self.create_model_results_page()

    def create_upload_page(self):
        """Page 1: Upload CSV File"""
        for widget in self.page_upload.winfo_children():
            widget.destroy()
        self.page_upload.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_upload,
            text="Upload CSV File",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=30, padx=20)

        upload_btn = ctk.CTkButton(
            self.page_upload,
            text="Select File",
            command=self.load_csv,
            width=300,
            fg_color="#4CAF50",
            font=("Helvetica", 16)
        )
        upload_btn.grid(row=1, column=0, pady=20)

        self.page_upload.pack(fill="both", expand=True)

    def create_data_view_page(self):
        """Page 2: Display Uploaded CSV Data"""
        for widget in self.page_data_view.winfo_children():
            widget.destroy()
        self.page_data_view.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_data_view,
            text="Uploaded CSV Data",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        self.frame_table_original = ctk.CTkFrame(self.page_data_view)
        self.frame_table_original.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_data_view.rowconfigure(1, weight=1)

        next_btn = ctk.CTkButton(
            self.page_data_view,
            text="Next: Select Label and Features",
            command=self.show_page_column_select,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        next_btn.grid(row=2, column=0, pady=20)

    def create_column_selection_page(self):
        """Page 3: Select Label and Feature Columns"""
        for widget in self.page_column_select.winfo_children():
            widget.destroy()
        self.page_column_select.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_column_select,
            text="Select Label and Features",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        # Frame for label selection
        self.label_frame = ctk.CTkFrame(self.page_column_select)
        self.label_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        # Frame for feature selection
        self.column_frame = ctk.CTkFrame(self.page_column_select)
        self.column_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.page_column_select.rowconfigure(2, weight=1)

        self.label_var = ttk.StringVar()

        # Create label selection dropdown
        label_label = ctk.CTkLabel(
            self.label_frame,
            text="Select Label Column:",
            font=("Arial", 18, "bold")
        )
        label_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        self.label_menu = ttk.Combobox(
            self.label_frame,
            state="readonly",
            textvariable=self.label_var,
            font=("Arial", 14)
        )
        self.label_menu.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.label_menu.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Process Data button
        process_btn = ctk.CTkButton(
            self.page_column_select,
            text="Process Data",
            command=self.process_data,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        process_btn.grid(row=3, column=0, pady=20)

    def create_results_page(self):
        """Page 4: Show Processed & Original Data with Navigation Options"""
        for widget in self.page_results.winfo_children():
            widget.destroy()
        self.page_results.grid_columnconfigure(0, weight=1)

        nav_frame = ctk.CTkFrame(self.page_results)
        nav_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        nav_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_original = ctk.CTkButton(
            nav_frame,
            text="View Original Data",
            command=lambda: self.show_table(self.df_original, "Original Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_original.grid(row=0, column=0, padx=10)

        self.btn_processed = ctk.CTkButton(
            nav_frame,
            text="View Processed Data",
            command=lambda: self.show_table(self.df_processed, "Processed Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_processed.grid(row=0, column=1, padx=10)

        # Button to jump to classification model page
        self.btn_model = ctk.CTkButton(
            nav_frame,
            text="Run Classification",
            command=lambda: self.show_page(self.page_model_select),
            width=250,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_model.grid(row=0, column=2, padx=10)

        self.result_table_frame = ctk.CTkFrame(self.page_results)
        self.result_table_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_results.rowconfigure(1, weight=1)

    def create_model_selection_page(self):
        """Page 5: Classification Model Processing Page"""
        for widget in self.page_model_select.winfo_children():
            widget.destroy()
        self.page_model_select.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_model_select,
            text="Classification Model",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        info = ctk.CTkLabel(
            self.page_model_select,
            text="Click the button below to train classifier on the processed data.",
            font=("Helvetica", 16)
        )
        info.grid(row=1, column=0, pady=10, padx=20)

        run_model_btn = ctk.CTkButton(
            self.page_model_select,
            text="Run Classification",
            command=self.run_classification,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        run_model_btn.grid(row=2, column=0, pady=20)

        back_btn = ctk.CTkButton(
            self.page_model_select,
            text="Back to Results",
            command=lambda: self.show_page(self.page_results),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=3, column=0, pady=10)

    def create_model_results_page(self):
        """Page 6: Display Classification Model Results and Allow Model Dumping"""
        for widget in self.page_model_results.winfo_children():
            widget.destroy()
        self.page_model_results.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_model_results,
            text="Model Results and Dumping",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        # Frame for displaying the global_model_results DataFrame
        self.model_results_frame = ctk.CTkFrame(self.page_model_results)
        self.model_results_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_model_results.rowconfigure(1, weight=1)

        # Button to dump selected models
        self.dump_btn = ctk.CTkButton(
            self.page_model_results,
            text="Dump Selected Models",
            command=self.dump_selected_models,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        self.dump_btn.grid(row=2, column=0, pady=10)

        # Back button
        back_btn = ctk.CTkButton(
            self.page_model_results,
            text="Back to Classification",
            command=lambda: self.show_page(self.page_model_select),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=3, column=0, pady=10)

    def show_page(self, page):
        """Hide all pages and display the selected page."""
        for p in [
            self.page_upload, self.page_data_view, self.page_column_select,
            self.page_results, self.page_model_select, self.page_model_results
        ]:
            p.pack_forget()
            p.grid_forget()
        page.pack(fill="both", expand=True)

    def load_csv(self):
        """Load CSV file and display its contents."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            self.display_table(self.df_original, self.frame_table_original, "Original Data")
            self.show_page(self.page_data_view)
            # Populate the label dropdown options
            self.label_menu['values'] = list(self.df_original.columns)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def show_page_column_select(self, event=None):
        """Populate label and feature selection widgets."""
        for widget in self.label_frame.winfo_children():
            widget.destroy()
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        # Recreate label selection dropdown
        label_label = ctk.CTkLabel(
            self.label_frame,
            text="Select Label Column:",
            font=("Arial", 18, "bold")
        )
        label_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")
        self.label_menu = ttk.Combobox(
            self.label_frame,
            state="readonly",
            textvariable=self.label_var,
            font=("Arial", 14),
            values=list(self.df_original.columns)
        )
        self.label_menu.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.label_menu.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Create feature selection checkboxes
        self.update_feature_selection()

        self.show_page(self.page_column_select)

    def update_feature_selection(self, event=None):
        """Update feature selection based on chosen label."""
        label_column = self.label_var.get()
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        self.feature_vars = []
        remaining_columns = [col for col in self.df_original.columns if col != label_column]

        for idx, col in enumerate(remaining_columns):
            var = ttk.IntVar()
            chk = ttk.Checkbutton(
                self.column_frame,
                text=col,
                variable=var
            )
            chk.grid(row=idx, column=0, sticky="w", padx=20, pady=5)
            self.feature_vars.append((col, var))

    def process_data(self):
        """Process selected label and features and automatically save preprocessed data."""
        label_column = self.label_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get() == 1]

        if not label_column or not selected_features:
            messagebox.showerror("Error", "Please select a label column and at least one feature!")
            return

        try:
            data_subset = self.df_original[selected_features + [label_column]].copy()
            self.df_processed = self.preprocess_data(data_subset)
            self.btn_original.configure(state="normal")
            self.btn_processed.configure(state="normal")
            self.btn_model.configure(state="normal")
            self.df_processed.to_csv("preprocessed_data.csv", index=False)
            messagebox.showinfo("Download", "Preprocessed data saved as 'preprocessed_data.csv'.")
            self.show_page(self.page_results)
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {e}")

    def preprocess_data(self, data):
        """Preprocessing: fill missing values, standardize numeric columns, encode categoricals."""
        try:
            # Fill missing numeric values with geometric mean (if possible)
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing = data[col].dropna()
                positive = non_missing[non_missing > 0]
                if not positive.empty:
                    geom_mean = gmean(positive)
                    data.loc[data[col].isna(), col] = geom_mean

            # Standardize numeric columns (excluding the label column)
            label_column = self.label_var.get()  # Get the label column
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

            # Exclude the label column from numeric_cols
            if label_column in numeric_cols:
                numeric_cols = numeric_cols.drop(label_column)

            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            joblib.dump(scaler, "scaler.pkl")

            # Encode categorical columns
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
        """Display a data table inside the provided frame."""
        for widget in frame.winfo_children():
            widget.destroy()

        title_label = ctk.CTkLabel(frame, text=title, font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        container = ctk.CTkFrame(frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        tree = tk_tt.Treeview(container, columns=df.columns.tolist(), show="headings")
        vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for col in df.columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)

        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def show_table(self, df, title):
        """Show a data table in the Results page."""
        self.display_table(df, self.result_table_frame, title)

    def run_classification(self):
        """Run classification models and display results."""
        label_column = self.label_var.get()
        if label_column not in self.df_processed.columns:
            messagebox.showerror("Error", "Label column not found in processed data.")
            return

        try:
            X = self.df_processed.drop(columns=[label_column])
            y = self.df_processed[label_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

            # Train all models
            fcp.train_svc(X_train, y_train, X_test, y_test)
            fcp.train_knn(X_train, y_train, X_test, y_test)
            fcp.train_random_forest(X_train, y_train, X_test, y_test)
            fcp.train_decision_tree(X_train, y_train, X_test, y_test)
            fcp.train_logistic_regression(X_train, y_train, X_test, y_test)

            # Display the global_model_results DataFrame
            self.global_model_results = fcp.global_model_results

            # Debugging: Print the contents of global_model_results
            print("Global Model Results:")
            print(self.global_model_results)

            # Display the results
            self.display_model_results(self.global_model_results)

            # Show the model results page
            self.show_page(self.page_model_results)
        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {e}")

    def display_model_results(self, global_model_results):
        """Display the global_model_results DataFrame in a table with checkboxes and row numbers."""
        for widget in self.model_results_frame.winfo_children():
            widget.destroy()

        # Debugging: Check if the DataFrame is empty
        if global_model_results.empty:
            print("Warning: global_model_results is empty!")
            return

        # Create a container frame for the Treeview
        container = ctk.CTkFrame(self.model_results_frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Add a "Row" column and a "Select" column to the Treeview
        columns = ["Row", "Select"] + list(global_model_results.columns)
        tree = tk_tt.Treeview(container, columns=columns, show="headings")
        vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Add columns to the Treeview
        for col in columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=100 if col == "Row" else 150)

        # Enable grid lines for better readability
        tree.configure(style="Treeview")

        # Add padding to the Treeview cells
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)  # Increase row height for better spacing
        style.configure("Treeview.Heading", font=("Arial", 12, "bold"))  # Style for headings
        style.configure("Treeview.Cell", padding=(10, 5))  # Add padding to cells

        # Add rows to the Treeview and store checkbox variables
        self.model_selection_vars = {}
        for idx, row in global_model_results.iterrows():
            # Insert the row data into the Treeview with row numbers and a placeholder for the checkbox
            values = [idx + 1, ""] + list(row)  # Add row number and an empty string for the checkbox column
            tree.insert("", "end", values=values)

            # Create a checkbox and place it over the "Select" column
            var = ttk.IntVar()
            chk = ttk.Checkbutton(container, variable=var)
            chk.place(x=120, y=35 + idx * 25)  # Adjust x and y to align with the Treeview rows
            self.model_selection_vars[row["model_name"]] = var

        # Bind the Treeview's scroll event to update checkbox positions
        def update_checkbox_positions(event):
            for idx, (model_name, var) in enumerate(self.model_selection_vars.items()):
                chk = ttk.Checkbutton(container, variable=var)
                chk.place(x=120, y=35 + idx * 25)  # Adjust x and y to align with the Treeview rows

        tree.bind("<Configure>", update_checkbox_positions)
        tree.bind("<MouseWheel>", update_checkbox_positions)

    def dump_selected_models(self):
        """Display metrics of selected models and ask for confirmation before dumping."""
        try:
            # Get the selected models
            selected_models = [
                model_name for model_name, var in self.model_selection_vars.items() if var.get() == 1
            ]

            if not selected_models:
                messagebox.showerror("Error", "Please select at least one model to dump!")
                return

            # Prepare the output message with metrics of selected models
            output_message = "Selected Models and Metrics:\n\n"
            for idx, model_name in enumerate(selected_models, start=1):
                model_info = self.global_model_results[self.global_model_results["model_name"] == model_name].iloc[0]
                algorithm = model_info["algorithm"]
                accuracy = model_info["accuracy"]
                f1_score = model_info["f1_score"]

                output_message += (
                    f"{idx}. Model: {model_name}\n"
                    f"   Algorithm: {algorithm}\n"
                    f"   Accuracy: {accuracy:.4f}\n"
                    f"   F1 Score: {f1_score:.4f}\n\n"
                )

            # Ask for confirmation before dumping
            confirm = messagebox.askyesno(
                "Confirm Dump",
                f"{output_message}\nDo you want to dump these models?"
            )

            if confirm:
                # Dump the selected models
                for model_name in selected_models:
                    model_info = self.global_model_results[self.global_model_results["model_name"] == model_name].iloc[
                        0]
                    model_filename = model_info["model_file"]
                    params = model_info["hyperparameters"]
                    algorithm = model_info["algorithm"]

                    # Initialize the model based on the algorithm
                    if algorithm == "RandomForest":
                        model = RandomForestClassifier(**params, random_state=42)
                    elif algorithm == "DecisionTree":
                        model = DecisionTreeClassifier(**params, random_state=42)
                    elif algorithm == "KNN":
                        model = KNeighborsClassifier(**params)
                    elif algorithm == "SVC":
                        model = SVC(**params)
                    elif algorithm == "Logistic Regression":
                        model = LogisticRegression(**params)
                    else:
                        print(f"Unknown algorithm: {algorithm}. Skipping.")
                        continue

                    # Train the model on the full dataset
                    X = self.df_processed.drop(columns=[self.label_var.get()])
                    y = self.df_processed[self.label_var.get()]
                    model.fit(X, y)

                    # Save the model to file
                    joblib.dump(model, model_filename)

                # Show success message
                messagebox.showinfo("Success", "Selected models dumped successfully!")


                # Assign the processed DataFrame to df2
                df2 = pd.concat([X, y], axis=1)

                # Debugging: Check the type and contents of df2
                # print(f"Type of df2: {type(df2)}")
                # print(f"Columns in df2: {df2.columns}")
                # print(f"First few rows of df2:\n{df2.head()}")

            def correlation_matrix(df2):
                return df2.corr()
                # Compute correlation matrix
                # corr_matrix = df2.corr()

            # Identify feature pairs with strong correlation
            def get_filtered_feature_pairs(corr_matrix, x_min=-0.5, x_max=1, y_min=0.5, y_max=1):
                feature_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if x_min <= corr_value <= x_max and y_min <= corr_value <= y_max:
                            feature_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
                return feature_pairs

            # Function to display the current plot
            def display_plot():
                global current_plot, fig, ax, canvas

                # Clear the previous figure
                for widget in frame.winfo_children():
                    widget.destroy()

                fig, ax = plt.subplots(figsize=(8, 6))

                if current_plot == 0:
                    # Show correlation heatmap
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title("Correlation Heatmap")
                else:
                    # Show scatter plot for the current feature pair
                    feature_x, feature_y, corr_value = feature_pairs[current_plot - 1]
                    sns.scatterplot(x=df2[feature_x], y=df2[feature_y], ax=ax)
                    ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y} (Corr: {corr_value:.2f})")
                    ax.set_xlabel(feature_x)
                    ax.set_ylabel(feature_y)

                # Embed the Matplotlib figure in the Tkinter window
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

            # Function to handle "Next" button
            def next_plot():
                global current_plot
                if current_plot < len(feature_pairs):  # Limit to available plots
                    current_plot += 1
                display_plot()

            # Function to handle "Previous" button
            def prev_plot():
                global current_plot
                if current_plot > 0:
                    current_plot -= 1
                display_plot()

            # Load dataset
            # file_path = "processed_data7.csv"  # Replace with your actual file path
            # df = load_data(file_path)

            # Compute correlation matrix
            corr_matrix = correlation_matrix(df2)

            # Get feature pairs that meet correlation conditions
            feature_pairs = get_filtered_feature_pairs(corr_matrix)

            # Initialize Tkinter window
            root = tk.Tk()
            root.title("Correlation Analysis")
            root.geometry("900x700")

            # Frame to hold the plot
            frame = tk.Frame(root)
            frame.pack()

            # Buttons for navigation
            button_frame = tk.Frame(root)
            button_frame.pack(side=tk.BOTTOM, pady=20)

            btn_prev = tk.Button(button_frame, text="Previous", command=prev_plot)
            btn_prev.pack(side=tk.LEFT, padx=10)

            btn_next = tk.Button(button_frame, text="Next", command=next_plot)
            btn_next.pack(side=tk.RIGHT, padx=10)

            # Start with the heatmap
            current_plot = 0
            display_plot()

            # Run Tkinter main loop
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Error during model dumping: {e}")

    # def run_regression(self):
    #     """Run classification models and display results."""
    #     label_column = self.label_var.get()
    #     if label_column not in self.df_processed.columns:
    #         messagebox.showerror("Error", "Label column not found in processed data.")
    #         return
    #
    #     try:
    #         X = self.df_processed.drop(columns=[label_column])
    #         y = self.df_processed[label_column]
    #
    #         # Split the data into training and testing sets
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    #
    #         # Train all models
    #         # fcp.train_svc(X_train, y_train, X_test, y_test)
    #         # fcp.train_knn(X_train, y_train, X_test, y_test)
    #         # fcp.train_random_forest(X_train, y_train, X_test, y_test)
    #         # fcp.train_decision_tree(X_train, y_train, X_test, y_test)
    #         # fcp.train_logistic_regression(X_train, y_train, X_test, y_test)
    #         frp.train_svr()
    #         frp.train_linear_regression()
    #         # Display the global_model_results DataFrame
    #         self.global_model_results = frp.global_model_results
    #
    #         # Debugging: Print the contents of global_model_results
    #         print("Global Model Results:")
    #         print(self.global_model_results)
    #
    #         # Display the results
    #         self.display_model_results(self.global_model_results)
    #
    #         # Show the model results page
    #         self.show_page(self.page_model_results)
    #     except Exception as e:
    #         messagebox.showerror("Error", f"Error during classification: {e}")
    #
    # def display_model_results(self, global_model_results):
    #     """Display the global_model_results DataFrame in a table with checkboxes and row numbers."""
    #     for widget in self.model_results_frame.winfo_children():
    #         widget.destroy()
    #
    #     # Debugging: Check if the DataFrame is empty
    #     if global_model_results.empty:
    #         print("Warning: global_model_results is empty!")
    #         return
    #
    #     # Create a container frame for the Treeview
    #     container = ctk.CTkFrame(self.model_results_frame)
    #     container.pack(fill="both", expand=True, padx=10, pady=10)
    #
    #     # Add a "Row" column and a "Select" column to the Treeview
    #     columns = ["Row", "Select"] + list(global_model_results.columns)
    #     tree = tk_tt.Treeview(container, columns=columns, show="headings")
    #     vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
    #     hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
    #     tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    #
    #     tree.grid(row=0, column=0, sticky="nsew")
    #     vsb.grid(row=0, column=1, sticky="ns")
    #     hsb.grid(row=1, column=0, sticky="ew")
    #     container.grid_rowconfigure(0, weight=1)
    #     container.grid_columnconfigure(0, weight=1)
    #
    #     # Add columns to the Treeview
    #     for col in columns:
    #         tree.heading(col, text=col, anchor="center")
    #         tree.column(col, anchor="center", width=100 if col == "Row" else 150)
    #
    #     # Enable grid lines for better readability
    #     tree.configure(style="Treeview")
    #
    #     # Add padding to the Treeview cells
    #     style = ttk.Style()
    #     style.configure("Treeview", rowheight=25)  # Increase row height for better spacing
    #     style.configure("Treeview.Heading", font=("Arial", 12, "bold"))  # Style for headings
    #     style.configure("Treeview.Cell", padding=(10, 5))  # Add padding to cells
    #
    #     # Add rows to the Treeview and store checkbox variables
    #     self.model_selection_vars = {}
    #     for idx, row in global_model_results.iterrows():
    #         # Insert the row data into the Treeview with row numbers and a placeholder for the checkbox
    #         values = [idx + 1, ""] + list(row)  # Add row number and an empty string for the checkbox column
    #         tree.insert("", "end", values=values)
    #
    #         # Create a checkbox and place it over the "Select" column
    #         var = ttk.IntVar()
    #         chk = ttk.Checkbutton(container, variable=var)
    #         chk.place(x=120, y=35 + idx * 25)  # Adjust x and y to align with the Treeview rows
    #         self.model_selection_vars[row["model_name"]] = var
    #
    #     # Bind the Treeview's scroll event to update checkbox positions
    #     def update_checkbox_positions(event):
    #         for idx, (model_name, var) in enumerate(self.model_selection_vars.items()):
    #             chk = ttk.Checkbutton(container, variable=var)
    #             chk.place(x=120, y=35 + idx * 25)  # Adjust x and y to align with the Treeview rows
    #
    #     tree.bind("<Configure>", update_checkbox_positions)
    #     tree.bind("<MouseWheel>", update_checkbox_positions)
    #
    # def dump_selected_models(self):
    #     """Display metrics of selected models and ask for confirmation before dumping."""
    #     try:
    #         # Get the selected models
    #         selected_models = [
    #             model_name for model_name, var in self.model_selection_vars.items() if var.get() == 1
    #         ]
    #
    #         if not selected_models:
    #             messagebox.showerror("Error", "Please select at least one model to dump!")
    #             return
    #
    #         # Prepare the output message with metrics of selected models
    #         output_message = "Selected Models and Metrics:\n\n"
    #         for idx, model_name in enumerate(selected_models, start=1):
    #             model_info = self.global_model_results[self.global_model_results["model_name"] == model_name].iloc[0]
    #             algorithm = model_info["algorithm"]
    #             accuracy = model_info["accuracy"]
    #             # r2_score = model_info["r2_score"]
    #             # mse = model_info["mse"]
    #             # rmse = model_info["rmse"]
    #             # mae = model_info["mae"]
    #             # explained_variance = model_info["explained_variance"]
    #             max_error = model_info["max_error"]
    #
    #
    #             output_message += (
    #                 f"{idx}. Model: {model_name}\n"
    #                 f"   Algorithm: {algorithm}\n"
    #                 f"   Accuracy: {accuracy:.4f}\n"
    #                 f"   r2 Score: {r2_score:.4f}\n"
    #                 f" rmse: {rmse:.4f}\n"
    #                 f"mae: {mae:.4f}\n"
    #                 f":explained_variance: {explained_variance:.4f}\n"
    #                 f"max_error: {max_error:.2f}\n\n"
    #             )
    #
    #         # Ask for confirmation before dumping
    #         confirm = messagebox.askyesno(
    #             "Confirm Dump",
    #             f"{output_message}\nDo you want to dump these models?"
    #         )
    #
    #         if confirm:
    #             # Dump the selected models
    #             for model_name in selected_models:
    #                 model_info = self.global_model_results[self.global_model_results["model_name"] == model_name].iloc[
    #                     0]
    #                 model_filename = model_info["model_file"]
    #                 params = model_info["hyperparameters"]
    #                 algorithm = model_info["algorithm"]
    #
    #                 # Initialize the model based on the algorithm
    #                 # if algorithm == "RandomForest":
    #                 #     model = RandomForestClassifier(**params, random_state=42)
    #                 # elif algorithm == "DecisionTree":
    #                 #     model = DecisionTreeClassifier(**params, random_state=42)
    #                 # elif algorithm == "KNN":
    #                 #     model = KNeighborsClassifier(**params)
    #                 # elif algorithm == "SVC":
    #                 #     model = SVC(**params)
    #                 # elif algorithm == "Logistic Regression":
    #                 #     model = LogisticRegression(**params)
    #                 if algorithm == "LinearRegression":
    #                     model = LinearRegression(**params, random_state=42)
    #                 elif algorithm == "SVR":
    #                     model = SVR(**params)
    #                 else:
    #                     print(f"Unknown algorithm: {algorithm}. Skipping.")
    #                     continue


        # except Exception as e:
        #     messagebox.showerror("Error", f"Error during model dumping: {e}")

if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVProcessorApp(root)
    root.mainloop()
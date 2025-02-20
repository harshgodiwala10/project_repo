import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Switcher")

        # Create Frame for Graph
        self.frame = tk.Frame(root)
        self.frame.pack()

        # Buttons
        self.btn_load = tk.Button(root, text="Load Data", command=self.load_data)
        self.btn_prev = tk.Button(root, text="Previous", command=self.show_scatter, state=tk.DISABLED)
        self.btn_next = tk.Button(root, text="Next", command=self.show_heatmap, state=tk.DISABLED)

        self.btn_load.pack(side=tk.TOP, padx=10, pady=5)
        self.btn_prev.pack(side=tk.LEFT, padx=10, pady=5)
        self.btn_next.pack(side=tk.RIGHT, padx=10, pady=5)

        # Initialize Graph
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()

        self.data = None

    def load_data(self):
        """ Load CSV file and preprocess data """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        # Load dataset
        self.data = pd.read_csv(file_path)

        # Preprocessing
        self.data = self.data.select_dtypes(include=[np.number])  # Keep only numerical columns
        self.data = self.data.dropna()  # Remove missing values
        scaler = MinMaxScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])  # Scale values

        # Enable Next Button
        self.btn_next.config(state=tk.NORMAL)

        # Show Scatter Plot
        self.show_scatter()

    def show_heatmap(self):
        """ Display Heatmap """
        if self.data is None:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        sns.heatmap(self.data.corr(), cmap="coolwarm", annot=True, ax=ax)  # Show correlation matrix

        ax.set_title("Heatmap")
        self.canvas.draw()

        # Disable Next button
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.DISABLED)

    def show_scatter(self):
        """ Display Scatter Plot """
        if self.data is None or self.data.shape[1] < 2:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        x_col, y_col = self.data.columns[:2]  # Select first two numeric columns
        ax.scatter(self.data[x_col], self.data[y_col], color='blue')
        ax.set_title("Scatter Plot")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        self.canvas.draw()

        # Disable Previous button
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL)




# Run the Tkinter App
root = tk.Tk()
app = GraphApp(root)
root.mainloop()

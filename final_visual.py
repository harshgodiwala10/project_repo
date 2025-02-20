import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Compute correlation matrix
def correlation_matrix(df):
    return df.corr()

# Identify feature pairs with strong correlation
def get_filtered_feature_pairs(corr_matrix, x_min=-0.6, x_max=1, y_min=0.6, y_max=1):
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
        sns.scatterplot(x=df[feature_x], y=df[feature_y], ax=ax)
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

# Load dataset4
file_path = "C:/Users/Harsh/OneDrive/Desktop/python practice/teleCust1000t.csv"  # Replace with your actual file path
df = load_data(file_path)

# Compute correlation matrix
corr_matrix = correlation_matrix(df)

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

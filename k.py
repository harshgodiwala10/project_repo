import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')  # Works well with Wayland and X11
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Load the dataset
def load_data(file_path):
    df = pd.read_csv('C:/Users/Harsh/OneDrive/Desktop/python practice/processed_data.csv')
    return df


# Compute correlation matrix
def correlation_matrix(df):
    return df.corr()


# Plot heatmap
def plot_correlation_heatmap(corr_matrix):
    """
    Plot a heatmap for the correlation matrix.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show(block=False)  # Allow the script to continue running



# Identify feature pairs based on correlation range
def get_filtered_feature_pairs(corr_matrix, x_min=-0.6, x_max=1, y_min=0.6, y_max=1):
    filtered_pairs = [
        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
        for i in range(len(corr_matrix.columns))
        for j in range(i + 1, len(corr_matrix.columns))
        if x_min <= corr_matrix.iloc[i, j] <= x_max and y_min <= corr_matrix.iloc[i, j] <= y_max
    ]
    return filtered_pairs


# Global variables
current_index = 0
feature_pairs = []
df = None
ax = None
fig = None


# Update scatter plot
def update_scatter_plot(event):
    global current_index, feature_pairs, df, ax

    if feature_pairs:
        feature_x, feature_y, corr_value = feature_pairs[current_index]
        ax.clear()  # Clear only the axes, not the figure
        sns.scatterplot(x=df[feature_x], y=df[feature_y], ax=ax)
        ax.set_title(f'Scatter Plot: {feature_x} vs {feature_y} (Corr: {corr_value:.2f})')
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        plt.draw()


# Button callbacks
def next_scatter(event):
    global current_index
    if current_index < len(feature_pairs) - 1:
        current_index += 1
        update_scatter_plot(event)


def previous_scatter(event):
    global current_index
    if current_index > 0:
        current_index -= 1
        update_scatter_plot(event)


# Show scatter plots one by one
def plot_scatter_plots(df_input, pairs):
    global feature_pairs, df, fig, ax
    df = df_input
    feature_pairs = pairs

    if feature_pairs:
        print("\nGenerating scatter plots one by one...")

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Create buttons
        ax_next = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_prev = plt.axes([0.59, 0.05, 0.1, 0.075])
        btn_next = Button(ax_next, 'Next')
        btn_prev = Button(ax_prev, 'Previous')

        # Attach callbacks
        btn_next.on_clicked(next_scatter)
        btn_prev.on_clicked(previous_scatter)

        # Plot the first scatter plot
        update_scatter_plot(None)
        plt.show()
    else:
        print("\nNo feature pairs meet the specified correlation range.")


# Main function
def main(file_path):
    global df
    df = load_data(file_path)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    corr_matrix = correlation_matrix(df)
    plot_correlation_heatmap(corr_matrix)

    feature_pairs = get_filtered_feature_pairs(corr_matrix)
    plot_scatter_plots(df, feature_pairs)


# Run the script
if __name__ == "__main__":
    file_path = 'processed_data3.csv'  # Replace with your actual file path
    main(file_path)

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Load the dataset
def load_data(file_path):
    """
    Load the dataset from the provided file path.
    """
    df = pd.read_csv('C:/Users/Harsh/OneDrive/Desktop/python practice/processed_data.csv')
    return df


# Perform correlation analysis
def correlation_matrix(df):
    """
    Calculate the correlation matrix of the dataset.
    """
    return df.corr()


# Identify feature pairs based on correlation range
def get_filtered_feature_pairs(corr_matrix, x_min=-0.6, x_max=1, y_min=0.6, y_max=1):
    """
    Identify feature pairs where the correlation value is in the specified range.
    """
    filtered_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):  # Avoid duplicate pairs
            corr_value = corr_matrix.iloc[i, j]
            if x_min <= corr_value <= x_max and y_min <= corr_value <= y_max:
                filtered_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
    return filtered_pairs


# Button function to go to the next plot
def next_plot(event, plots, current_index, ax, df, corr_matrix):
    """
    Function to go to the next plot when 'Next' is clicked.
    """
    if current_index[0] < len(plots) - 1:
        current_index[0] += 1
        ax.clear()
        plot_current(plots, current_index[0], ax, df, corr_matrix)
        plt.draw()


# Button function to go to the previous plot
def previous_plot(event, plots, current_index, ax, df, corr_matrix):
    """
    Function to go to the previous plot when 'Previous' is clicked.
    """
    if current_index[0] > 0:
        current_index[0] -= 1
        ax.clear()
        plot_current(plots, current_index[0], ax, df, corr_matrix)
        plt.draw()


# Plot the current plot based on index
def plot_current(plots, idx, ax, df, corr_matrix):
    """
    Plot the current plot based on index.
    """
    plot_type, *params = plots[idx]
    if plot_type == "heatmap":
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
    elif plot_type == "scatter":
        feature_x, feature_y, corr_value = params
        sns.scatterplot(x=df[feature_x], y=df[feature_y], ax=ax)
        ax.set_title(f'{feature_x} vs {feature_y} (Corr: {corr_value:.2f})')

        # Remove scale (ticks)
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove side bars (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")


# Show interactive plots with navigation
def plot_all_graphs_interactive(df, corr_matrix, feature_pairs):
    """
    Display heatmap and scatter plots interactively with next and previous buttons.
    """
    plots = [("heatmap",)]  # Start with the heatmap plot

    # Add scatter plots to the plots list
    for feature_x, feature_y, corr_value in feature_pairs:
        plots.append(("scatter", feature_x, feature_y, corr_value))

    current_index = [0]  # To keep track of the current plot

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_current(plots, current_index[0], ax, df, corr_matrix)

    # Add 'Next' and 'Previous' buttons
    ax_next = plt.axes([0.8, 0.02, 0.1, 0.05])  # Position for the Next button
    ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])  # Position for the Previous button

    btn_next = Button(ax_next, 'Next')
    btn_prev = Button(ax_prev, 'Previous')

    # Attach the button event handlers
    btn_next.on_clicked(lambda event: next_plot(event, plots, current_index, ax, df, corr_matrix))
    btn_prev.on_clicked(lambda event: previous_plot(event, plots, current_index, ax, df, corr_matrix))

    # Show the interactive plot
    plt.show()


# Main function
def main(file_path):
    # Load the data
    df = load_data(file_path)

    # Display first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check for any missing values in the dataset
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # Compute correlation matrix
    corr_matrix = correlation_matrix(df)

    # Get feature pairs that meet the correlation condition
    feature_pairs = get_filtered_feature_pairs(corr_matrix)

    # Show interactive plots with Next/Previous buttons
    print("\nGenerating interactive plots with Next/Previous buttons...")
    plot_all_graphs_interactive(df, corr_matrix, feature_pairs)


# Example usage
if __name__ == "__main__":
    file_path = 'Boston1.csv'  # Replace this with the actual file path
    main(file_path)

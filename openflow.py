import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

class InteractivePolygonGating:
    def __init__(self, dataframe, x_col, y_col, num_edges=4, log=False):
        # Initialize with dataframe and column names to be used for plotting.
        self.df = dataframe
        self.x_col = x_col
        self.y_col = y_col
        self.num_edges = num_edges  # Number of polygon vertices
        self.log = log              # Whether to use logarithmic transformation
        self.selected_data = None   # Data selected by the polygon
        
        # Extract x and y arrays from the dataframe columns
        self.x = self.df[self.x_col].values
        self.y = self.df[self.y_col].values
        
        # Store original data ranges for later use
        self.x_min = self.x.min()
        self.x_max = self.x.max()
        self.y_min = self.y.min()
        self.y_max = self.y.max()
        
        # Apply log transformation if required
        if self.log:
            self.x_temp = np.log10(self.x)
            self.y_temp = np.log10(self.y)
        else:
            self.x_temp = self.x
            self.y_temp = self.y

        # Turn on interactive mode for matplotlib
        plt.ion()

        # Create figure and axis for scatter plot
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(self.x_temp, self.y_temp, s=1, c='blue')

        # If using logarithmic scale, format tick labels to show original values in scientific notation
        if self.log:
            formatter = FuncFormatter(lambda val, pos: f"{10**val:.2e}")
            self.ax.xaxis.set_major_formatter(formatter)
            self.ax.yaxis.set_major_formatter(formatter)
            self.fig.canvas.draw_idle()

        # Set up the initial polygon vertices
        theta = np.linspace(0, 2*np.pi, self.num_edges, endpoint=False)
        # Determine radius as 20% of the smaller range from x_temp and y_temp
        xrange = self.x_temp.max() - self.x_temp.min()
        yrange = self.y_temp.max() - self.y_temp.min()
        radius = 0.2 * min(xrange, yrange)
        # Center the polygon on the data's mean position
        cx, cy = np.mean(self.x_temp), np.mean(self.y_temp)
        self.polygon_vertices = np.column_stack((radius*np.cos(theta) + cx,
                                                 radius*np.sin(theta) + cy))
        # Create a polygon patch using these vertices
        self.polygon = Polygon(self.polygon_vertices, closed=True,
                               linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.polygon)

        # Draw the vertices with black color
        self.vertex_scatter = self.ax.scatter(self.polygon_vertices[:,0],
                                              self.polygon_vertices[:,1],
                                              s=50, c='black', zorder=3)

        # Set axis limits with an extra padding related to the polygon size
        pad = 0.2 * radius
        self.ax.set_xlim(self.x_temp.min() - pad, self.x_temp.max() + pad)
        self.ax.set_ylim(self.y_temp.min() - pad, self.y_temp.max() + pad)

        # Prepare histograms on a separate figure
        self.fig_hist, (self.ax_histx, self.ax_histy) = plt.subplots(1, 2, figsize=(10, 4))
        if self.log:
            # Use formatter to display histograms in scientific notation for log scale
            formatter = FuncFormatter(lambda val, pos: f"{10**val:,.0f}")
            self.ax_histx.xaxis.set_major_formatter(formatter)
            self.ax_histy.xaxis.set_major_formatter(formatter)

        self.dragging_vertex = None  # This will track which polygon vertex is being dragged

        # Update the histograms initially
        self.update_histograms()

        # Connect mouse events for interactive behavior
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        plt.show(block=False)

    def on_press(self, event):
        # If click is outside the axis, ignore it
        if event.inaxes != self.ax:
            return
        # Get current axis limits for threshold calculation
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        # Compute view's diagonal length
        diagonal = np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0])
        # Set a threshold (2% of diagonal) for selecting a vertex
        threshold = 0.02 * diagonal

        # Loop through vertices to see if the click is close enough to a vertex
        for i, (vx, vy) in enumerate(self.polygon.get_xy()):
            if np.hypot(event.xdata - vx, event.ydata - vy) < threshold:
                self.dragging_vertex = i
                break

    def on_release(self, event):
        # When mouse is released, stop dragging a vertex
        self.dragging_vertex = None
        # Update histograms based on new polygon position
        self.update_histograms()

    def on_motion(self, event):
        # If no vertex is selected or event is outside axis, ignore the motion
        if self.dragging_vertex is None or event.inaxes != self.ax:
            return
        # Update the position of the dragged vertex to follow the mouse
        self.polygon_vertices[self.dragging_vertex] = [event.xdata, event.ydata]
        self.polygon.set_xy(self.polygon_vertices)
        # Update the scatter points for vertices with new positions
        self.vertex_scatter.set_offsets(self.polygon_vertices)
        self.fig.canvas.draw()

    def update_histograms(self):
        # Determine which data points are enclosed by the polygon
        enclosed_indices = []
        path = Path(self.polygon.get_xy())
        for i in range(len(self.x_temp)):
            if path.contains_point((self.x_temp[i], self.y_temp[i])):
                enclosed_indices.append(i)
        # Save the selected data subset for use elsewhere
        self.selected_data = self.df.iloc[enclosed_indices]

        # Clear and update histogram for x values from the enclosed data
        self.ax_histx.clear()
        self.ax_histx.hist(self.x[enclosed_indices], bins=50, color='blue', alpha=0.7)
        self.ax_histx.set_title('X values histogram')

        # Clear and update histogram for y values from the enclosed data
        self.ax_histy.clear()
        self.ax_histy.hist(self.y[enclosed_indices], bins=50, color='green', alpha=0.7)
        self.ax_histy.set_title('Y values histogram')

        # Set the overall title with count of selected items and redraw the figure
        self.fig_hist.suptitle(f"Number of selected items: {len(enclosed_indices)}")
        self.fig_hist.canvas.draw()

    def apply_gate(self, new_df, x_col=None, y_col=None):
        # Apply current polygon gate to a new dataframe
        if x_col is None: 
            x_col = self.x_col
        if y_col is None: 
            y_col = self.y_col

        # Extract x and y values from the new dataframe
        new_x = new_df[x_col].values
        new_y = new_df[y_col].values

        # Apply log transformation if needed
        if self.log:
            new_x_temp = np.log10(new_x)
            new_y_temp = np.log10(new_y)
        else:
            new_x_temp = new_x
            new_y_temp = new_y

        # Determine which points in the new dataframe are inside the polygon
        path = Path(self.polygon.get_xy())
        enclosed_indices = []
        for i in range(len(new_x_temp)):
            if path.contains_point((new_x_temp[i], new_y_temp[i])):
                enclosed_indices.append(i)

        # Return only the points that are inside the polygon
        return new_df.iloc[enclosed_indices]


# # Example usage:
# if __name__ == "__main__":
#     np.random.seed(0)
#     # Generate sample data using normal distribution
#     data = {
#         'x': np.random.normal(0.5, 0.1, 60000),
#         'y': np.random.normal(0.5, 0.1, 60000)
#     }
#     df = pd.DataFrame(data)

#     # Create an instance of InteractivePolygonGating with 5 vertices and logarithmic scale enabled.
#     gate1 = InteractivePolygonGating(df, 'x', 'y', num_edges=5, log=True)
#     # As you drag the polygon the histograms and scatter plot will update.


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class InteractiveHistogramThreshold:
    def __init__(self, data, threshold_channel, plot_channels, bins=50):
        """
        data: Pandas DataFrame containing all required channels.
        threshold_channel: The channel used for threshold selection.
        plot_channels: A list of channels to visualize from the selected data.
        bins: Number of bins for histogram.
        """
        self.data = data
        self.threshold_channel = threshold_channel
        # Ensure plot_channels is a list
        self.plot_channels = plot_channels if isinstance(plot_channels, list) else [plot_channels]
        self.bins = bins

        # Validate threshold channel
        if self.threshold_channel not in data.columns:
            raise ValueError(f"'{self.threshold_channel}' not found in the DataFrame columns.")

        # Validate plot channels
        for ch in self.plot_channels:
            if ch not in data.columns:
                raise ValueError(f"'{ch}' is not in the DataFrame columns.")

        self.selected_data = None

        # Main figure for threshold channel
        self.fig, self.ax = plt.subplots()
        self.hist_data, self.bin_edges, _ = self.ax.hist(
            self.data[self.threshold_channel],
            bins=self.bins,
            alpha=0.5,
            color='gray'
        )

        # Add vertical threshold lines around 20% and 80% by default
        self.lower_line = self.ax.axvline(
            self.bin_edges[int(len(self.bin_edges) * 0.2)],
            color='blue',
            linestyle='--'
        )
        self.upper_line = self.ax.axvline(
            self.bin_edges[int(len(self.bin_edges) * 0.8)],
            color='red',
            linestyle='--'
        )

        # Set up interactive events
        self.dragging_line = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Figure for plotting multiple channels
        n_channels = len(self.plot_channels)
        n_cols = n_channels
        n_rows = math.ceil(n_channels / n_cols)
        self.fig_selected, self.ax_selected = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        self.ax_selected = np.array(self.ax_selected).reshape(-1) if n_channels > 1 else [self.ax_selected]

        self.update_title()
        self.update_selected_data()

        # Show all figures non-blocking
        plt.show(block=False)

    def update_title(self):
        low = self.lower_line.get_xdata()[0]
        high = self.upper_line.get_xdata()[0]
        self.ax.set_title(
            f"Threshold channel '{self.threshold_channel}' in range [{low:.1f}, {high:.1f}]"
        )
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x = event.xdata
        dist_lower = abs(x - self.lower_line.get_xdata()[0])
        dist_upper = abs(x - self.upper_line.get_xdata()[0])
        # Threshold for detecting if we're close enough to a line
        threshold = (self.bin_edges[-1] - self.bin_edges[0]) * 0.02

        if dist_lower < threshold:
            self.dragging_line = self.lower_line
        elif dist_upper < threshold:
            self.dragging_line = self.upper_line

    def on_release(self, event):
        self.dragging_line = None
        self.update_selected_data()

    def on_motion(self, event):
        if self.dragging_line is None or event.xdata is None or event.inaxes != self.ax:
            return
        self.dragging_line.set_xdata([event.xdata, event.xdata])
        self.update_title()

    def update_selected_data(self):
        low = self.lower_line.get_xdata()[0]
        high = self.upper_line.get_xdata()[0]
        self.selected_data = self.data[
            (self.data[self.threshold_channel] >= low) &
            (self.data[self.threshold_channel] <= high)
        ]
        print(f"Number of selected items: {len(self.selected_data)}")
        self.plot_selected_channels()

    def plot_selected_channels(self):
        # Clear subplots and re-plot for each channel in plot_channels
        for ax in self.ax_selected:
            ax.clear()

        for i, ch in enumerate(self.plot_channels):
            if i < len(self.ax_selected):
                ax_ch = self.ax_selected[i]
                if not self.selected_data.empty:
                    ax_ch.hist(self.selected_data[ch], bins=self.bins, alpha=0.5, color='blue')
                    ax_ch.set_title(f"Selected Data: '{ch}'")
                else:
                    ax_ch.set_title(f"No data selected for '{ch}'")

        self.fig_selected.canvas.draw_idle()

# # Example usage:
# if __name__ == "__main__":
#     np.random.seed(42)
#     columns = ['channelA', 'channelB', 'channelC', 'channelD', 'channelE']
#     df = pd.DataFrame(
#         np.random.normal(loc=128, scale=25, size=(10000, len(columns))).clip(0, 255),
#         columns=columns
#     )

#     # threshold_channel is used to define the selection range
#     # plot_channels is a list of channels you want to see from the selected data
#     gating = InteractiveHistogramThreshold(
#         data=filtered_df,
#         threshold_channel='FL5-A',
#         plot_channels=['FL5-A','FL11-A']
#     )
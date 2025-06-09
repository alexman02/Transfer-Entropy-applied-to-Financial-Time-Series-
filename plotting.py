import numpy as np
import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ipywidgets import interact, Dropdown


def compute_measure_matrix(windows, measure_func, *measure_args, **measure_kwargs):
    """
    Generalized matrix computation for any pairwise measure
    
    Parameters:
        windows: iterable of (start_date, end_date, window_data)
        measure_func: function that takes two arrays and returns a scalar
        measure_kwargs: keyword arguments for measure_func
        
    Returns:
        measure_matrix: numpy array of shape (n_assets, n_assets, num_windows)
    """
    assets = windows[0][2].columns
    n = len(assets)
    num_windows = len(windows)
    measure_matrix = np.empty((n, n, num_windows))

    for t, (_, _, window_data) in enumerate(windows):
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if asset1 == asset2:
                    measure_matrix[i, j, t] = 0.0
                else:
                    measure_matrix[i, j, t] = measure_func(
                        window_data[asset1].values,
                        window_data[asset2].values,
                        *measure_args,
                        **measure_kwargs
                    )
        #print(f"Window {t+1}/{num_windows} computed")
    return measure_matrix

def measure_matrix_to_dict(measure_matrix, assets):
    """
    Convert 3D matrix to dictionary of time series
    
    Parameters:
        measure_matrix: array of shape (n, n, t)
        assets: list of asset names
        
    Returns:
        dict: {(asset1, asset2): [values]}
    """
    asset_idx = {asset: i for i, asset in enumerate(assets)}
    return {
        (a1, a2): measure_matrix[asset_idx[a1], asset_idx[a2], :].tolist()
        for a1, a2 in itertools.permutations(assets, 2)
    }

### Generalized Plotting Functions ###

def plot_measure_evolution(measure_dict, endpoints, assets, 
                         measure_name="measure", figsize=(12,6)):
    """
    Plot evolution of all pairwise measures over time
    
    Parameters:
        measure_dict: result from measure_matrix_to_dict
        endpoints: list of datetime objects for x-axis
        assets: list of asset names
        measure_name: name for y-axis label
    """
    plt.figure(figsize=figsize)
    asset_pairs = list(itertools.permutations(assets, 2))
    
    for pair in asset_pairs:
        plt.plot(endpoints, measure_dict[pair])
    
    plt.xticks(endpoints[::2], [d.year for d in endpoints[::2]], rotation=45)
    plt.ylabel(measure_name)
    plt.xlabel("End Date of Window")
    plt.tight_layout()
    plt.show()

def create_measure_heatmap(measure_matrix, assets, endpoints, 
                         measure_name="measure", colorscale="Viridis"):
    """
    Create interactive heatmap animation for any measure
    
    Parameters:
        measure_matrix: array from compute_measure_matrix
        assets: list of asset names
        endpoints: list of end dates for animation
        measure_name: name for title
        colorscale: plotly colorscale name
    """
    num_windows = measure_matrix.shape[2]
    
    # Create frames
    frames = [
        go.Frame(
            name=f"Window {t}",
            data=go.Heatmap(
                z=measure_matrix[:, :, t],
                x=assets,
                y=assets,
                colorscale=colorscale
            )
        )
        for t in range(num_windows)
    ]
    
    # Create slider steps
    slider_steps = [{
        "args": [[f"Window {t}"], {"frame": {"duration": 0}, "mode": "immediate"}],
        "label": str(endpoints[t].year),
        "method": "animate"
    } for t in range(num_windows)]
    
    # Create figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"{measure_name} Heatmap Over Time",
            sliders=[{
                "active": 0,
                "steps": slider_steps,
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Year up to: ", 
                    "xanchor": "right"},
                "pad": {"b": 10, "t": 50}
            }]
        )
    )
    return fig

def create_interactive_pair_plot(measure_dict, endpoints, assets, 
                                measure_name="measure"):
    """
    Create interactive widget for pairwise measure visualization
    """
    def plot_fn(asset1, asset2):
        plt.figure(figsize=(10,4))
        plt.plot(endpoints, measure_dict[(asset1, asset2)], marker='o')
        plt.title(f"{measure_name} from {asset1} to {asset2}")
        plt.xticks(endpoints[::2], [d.year for d in endpoints[::2]], rotation=45)
        plt.grid(True)
        plt.show()
    
    return interact(plot_fn, 
                   asset1=Dropdown(options=assets),
                   asset2=Dropdown(options=assets))

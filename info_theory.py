import numpy as np
import seaborn as sns
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

def entropy(X):
    _, num_counts = np.unique(X, return_counts=True, axis=0)
    p = num_counts / sum(num_counts)
    ent = -np.sum(p * np.log(p))
    return ent


def MI(X, Y):
    mi = entropy(X) + entropy(Y) - entropy(np.column_stack((X, Y)))
    return mi

def transfer_entropy_fast(X, Y, k, l=1):
    """
    Fast transfer entropy function
    TE_{X->Y} = H(Y^{+}| Y^{-}) - H(Y^{+}|Y^{-}, X^{-})
    """
    N = len(X)
    m = max(k, l)

    def sliding_window(arr, window_size):
        shape = (arr.size - window_size + 1, window_size)
        strides = (arr.strides[0], arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    
    x_windows = sliding_window(X, k)
    y_windows = sliding_window(Y, l)
    x_pasts = x_windows[m - k : N - k]
    y_pasts = y_windows[m - l : N - l]
    y_futures = sliding_window(Y, 1)[m : N]
    
    # Full joint: (X_past, Y_past, Y_future)
    full_joint = np.hstack((x_pasts, y_pasts, y_futures))
    # Joint for (Y_past, Y_future)
    y_joint = np.hstack((y_pasts, y_futures))
    
    H_full = entropy(full_joint)          # H(X_past, Y_past, Y_future)
    H_xy = entropy(np.hstack((x_pasts, y_pasts)))  # H(X_past, Y_past)
    H_yjoint = entropy(y_joint)             # H(Y_past, Y_future)
    H_y = entropy(y_pasts)                  # H(Y_past)
    
    H_yfuture_given_xy = H_full - H_xy        # H(Y_future | X_past, Y_past)
    H_yfuture_given_y = H_yjoint - H_y          # H(Y_future | Y_past)
    
    TE = H_yfuture_given_y - H_yfuture_given_xy
    return TE


def transfer_entropy(X, Y, k, l=1):
    N = len(X)
    m = max(k, l)

    def sliding_window(arr, window_size):
        shape = (arr.size - window_size + 1, window_size)  # Shape of output
        strides = (arr.strides[0], arr.strides[0])         # Strides (bytes to step)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    x_windows = sliding_window(X, k)      # shape: (N - k + 1, k)
    y_windows = sliding_window(Y, l)      # shape: (N - l + 1, l)
    # For each i in range(m, N), get X[i-k:i] -> index in sliding window is i - k:
    x_pasts = x_windows[m - k : N - k]
    y_pasts = y_windows[m - l : N - l]
    y_futures = sliding_window(Y, 1)[m : N]

    xyy = np.hstack((x_pasts, y_pasts, y_futures))

    xyy_unique, xyy_counts = np.unique(xyy, return_counts=True, axis=0)

    xy_unique, xy_counts = np.unique(xyy[:, :-1], return_counts=True, axis=0)
    yy_unique, yy_counts = np.unique(xyy[:, k:], return_counts=True, axis=0)
    y_unique, y_counts = np.unique(xyy[:, k:k+l], return_counts=True, axis=0)

    def counts_dict(unique, counts):
        return {tuple(event): count for event, count in zip(unique, counts)}

    xy_dict = counts_dict(xy_unique, xy_counts)
    yy_dict = counts_dict(yy_unique, yy_counts)
    y_dict = counts_dict(y_unique, y_counts)

    # Compute transfer entropy
    TE = 0
    for event, count in zip(xyy_unique, xyy_counts):
        y_future = event[-1]  
        y_past = event[k:k+l]  # y_past is the l past values of Y
        x_past = event[:k]  # x_past is the k past values of X

        # p(x_past, y_past, y_future)
        p_xyy = count / np.sum(xyy_counts)

        # p(y_future | x_past, y_past)
        xy_key = tuple(np.hstack((x_past, y_past)))  # Key for (x_past, y_past)
        condp_xy = count / xy_dict[xy_key]

        # p(y_future | y_past)
        yy_key_future = tuple(np.hstack((y_past, y_future)))  # Key for (y_past, y_future)
        yy_key_past = tuple(y_past)  # Key for y_past
        condp_y = yy_dict[yy_key_future] / y_dict[yy_key_past]

        # Compute the log term
        if condp_xy > 0 and condp_y > 0:
            log_term = np.log(condp_xy / condp_y)  
        else:
            log_term = 0  

        # Add to transfer entropy
        TE += p_xyy * log_term
    return TE

def transfer_entropy_vis(X, Y, k, l=1, probabilities=True, plot_diagnostics=False):
    """
    Compute the transfer entropy from X to Y with visualization capabilities.
    """
    N = len(X)
    m = max(k, l)

    # Numpy sliding window is faster than repeated slicing
    def sliding_window(arr, window_size):
        shape = (arr.size - window_size + 1, window_size)
        strides = (arr.strides[0], arr.strides[0])
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # Generate windows
    x_windows = sliding_window(X, k)
    y_windows = sliding_window(Y, l)
    x_pasts = x_windows[m - k : N - k]
    y_pasts = y_windows[m - l : N - l]
    y_futures = sliding_window(Y, 1)[m : N]

    # Combine into single array
    xyy = np.hstack((x_pasts, y_pasts, y_futures))

    # Get unique events and counts
    xyy_unique, xyy_counts = np.unique(xyy, return_counts=True, axis=0)
    xy_unique, xy_counts = np.unique(xyy[:, :-1], return_counts=True, axis=0)
    yy_unique, yy_counts = np.unique(xyy[:, k:], return_counts=True, axis=0)
    y_unique, y_counts = np.unique(xyy[:, k:k+l], return_counts=True, axis=0)

    # Create dictionaries for counts
    def counts_dict(unique, counts):
        return {tuple(event): count for event, count in zip(unique, counts)}
    
    xy_dict = counts_dict(xy_unique, xy_counts)
    yy_dict = counts_dict(yy_unique, yy_counts)
    y_dict = counts_dict(y_unique, y_counts)

    # Collect probabilities
    d = {
        'x_t': [],
        'y_t': [],
        'y_t+1': [],
        'p(x_t, y_t, y_t+1)': [],
        'p(y_t+1|x_t, y_t)': [],
        'p(y_t+1|y_t)': []
    }

    TE = 0
    total_counts = np.sum(xyy_counts)
    
    for event, count in zip(xyy_unique, xyy_counts):
        x_past = tuple(event[:k])
        y_past = tuple(event[k:k+l])
        y_future = event[-1]

        # p(x_past, y_past, y_future)
        p_xyy = count / total_counts
        
        # Get keys
        xy_key = tuple(np.hstack((x_past, y_past)))
        yy_key_future = tuple(np.hstack((y_past, y_future)))
        yy_key_past = tuple(y_past)
        
        # p(y_future | x_past, y_past)
        condp_xy = count / xy_dict[xy_key]

        # p(y_future | y_past)
        condp_y = yy_dict[yy_key_future] / y_dict[yy_key_past]

        # Store values
        d['x_t'].append(x_past)
        d['y_t'].append(y_past)
        d['y_t+1'].append(y_future)
        d['p(x_t, y_t, y_t+1)'].append(p_xyy)
        d['p(y_t+1|x_t, y_t)'].append(condp_xy)
        d['p(y_t+1|y_t)'].append(condp_y)

        # Calculate TE
        if condp_xy > 0 and condp_y > 0:
            TE += p_xyy * (np.log(condp_xy) - np.log(condp_y))

    d['log_term'] = np.log(d['p(y_t+1|x_t, y_t)']) - np.log(d['p(y_t+1|y_t)'])
    df = pd.DataFrame(d)

    if probabilities:
        """
        Assuming source and destination past history and delay are 1, we have
        TE_{X->Y}(1, 1, 1) = \sum p(y_t+1, y_t, x_t) log(p(y_t+1|y_t, x_t)/p(y_t+1|y_t))
        Sum is over possible triples (y_t+1, y_t, x_t). 
        Plot heatmaps of the joint and conditional distributions in the formula.
        """
        # Generate all possible events (to visualise the whole sample space)
        possible_x = np.unique(X)
        possible_y = np.unique(Y)
        
        all_events = []
        for x_past in product(possible_x, repeat=k):
            for y_past in product(possible_y, repeat=l):
                for y_future in possible_y:
                    all_events.append({
                        'x_t': x_past,
                        'y_t': y_past,
                        'y_t+1': y_future
                    })
        
        complete_df = pd.DataFrame(all_events)
        
        # Merge with calculated probabilities (events that happened in the data)
        merged_df = complete_df.merge(
            df, 
            on=['x_t', 'y_t', 'y_t+1'], 
            how='left'
        ).fillna(0)  # events that did not happen are filled with 0
        
        """
        For the third plot, the 0 values makes it appear that there is dependency on X_t.
        Fix this by precomputing p(y_t+1|y_t) independent of X_t and overwriting the 
        corresponding column of the dataframe.
        """
        # Precompute p(y_t+1|y_t) independent of X_t
        condp_y_dict = {}
        for yy_row, count in zip(yy_unique, yy_counts):  
            y_past = tuple(yy_row[:l])  
            y_future = yy_row[-1]       # Last element is y_future
            condp_y = count / y_dict[y_past]
            condp_y_dict[(y_past, y_future)] = condp_y

        # Replace p(y_t+1|y_t) column with precomputed values 
        merged_df['p(y_t+1|y_t)'] = merged_df.apply(
            lambda row: condp_y_dict.get((tuple(row['y_t']), row['y_t+1']), 0),
            axis=1
        )

    if plot_diagnostics:
        # Create plots
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        
        # Helper function to plot heatmaps
        def plot_heatmap(data, ax, title):
            if k == 1 and l == 1:
                # Convert tuples to scalar values
                data['x'] = data['x_t'].apply(lambda x: x[0])
                data['y'] = data['y_t'].apply(lambda y: y[0])
                pivot = data.pivot_table(
                    index='x', 
                    columns=['y', 'y_t+1'], 
                    values=title,
                    fill_value=0
                )
                sns.heatmap(pivot, ax=ax, cmap='viridis', annot=False, fmt=".2f")
                ax.set_title(title)
                ax.set_xlabel('(Y_t, Y_t+1)')
                ax.set_ylabel('X_t')
        
        # Plot each probability
        plot_heatmap(merged_df, axs[0], 'p(x_t, y_t, y_t+1)')
        plot_heatmap(merged_df, axs[1], 'p(y_t+1|x_t, y_t)')
        plot_heatmap(merged_df, axs[2], 'p(y_t+1|y_t)')
        plot_heatmap(merged_df, axs[3], 'log_term')

        plt.tight_layout()
        plt.show()

    return TE, merged_df


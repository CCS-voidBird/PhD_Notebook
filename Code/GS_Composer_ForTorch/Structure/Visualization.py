import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

def plot_loss_history(h, title,plot_name=None,checkpoint=0,numTrait=1):
    print("Plotting loss history...")
    hist_df = pd.DataFrame(h.history)
    hist_df['round'] = abs(checkpoint)
    hist_df['epoch'] = hist_df.index
    checkpoint = abs(checkpoint)
    try:
        history_record = pd.read_csv(plot_name+"_history.csv", sep="\t")
        history_record = history_record.append(hist_df)
        history_record.to_csv(plot_name+"_history.csv", sep="\t",index=False)
    except:
        hist_df.to_csv(plot_name+"_history.csv", sep="\t",index=False)


    plot_name_loss=plot_name+"_"+str(checkpoint)+"_loss.png"
    fig, axs = plt.subplots(2, 1)
    for i in range(numTrait):
        axs[0].plot(h.history['loss'][1:], label = "Train loss", color = "blue")
        axs[0].plot(h.history['val_loss'][1:], label = "Validation loss", color = "red")
        axs[0].set_ylabel("MSE")
        axs[1].plot(h.history['p_corr'][1:], label = "Train cor", color = "blue")
        axs[1].plot(h.history['val_p_corr'][1:], label = "Validation cor", color = "red")
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel("Pearson's Correlation")

    fig.suptitle(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name:
        plt.legend()
        plt.savefig(plot_name_loss)
        plt.close()

        #read history csv file from path
        

    else:
        return
      
def plot_correlation(predictions, observations, title,plot_name=None,checkpoint=0,numTrait=1):

    ###create clear correlation plot for predictions and observations

    print("Plotting predictions and observations..")
    hist_df = pd.DataFrame()
    hist_df['Individual'] = [x for x in range(len(predictions))]
    hist_df['Prediction'] = predictions
    hist_df['Observation'] = observations
    hist_df.sort_values(by='Observation', inplace=True)
    try:
        history_record = pd.read_csv(plot_name+title+"_correlation.csv", sep="\t")
        history_record = history_record.append(hist_df)
        #history_record.to_csv(plot_name+title+"_correlation.csv", sep="\t",index=False)
    except:
        hist_df.to_csv(plot_name+"_correlation.csv", sep="\t",index=False)


    plot_name_loss=plot_name+"_"+str(checkpoint)+"_correlation.png"
    fig = plt.figure()
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x=hist_df['Individual'],y=hist_df['Observation'], label = "Observation", color = "blue")
    axs.scatter(x=hist_df['Individual'],y=hist_df['Prediction'], label = "Prediction", color = "red")
    #axs[0].plot(hist_df['val_loss'][1:], label = "Validation loss", color = "red")
    axs.set_ylabel(" ")
    fig.suptitle(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name:
        plt.legend()
        plt.savefig(plot_name_loss)
        plt.close()
        

    else:
        return

    #plt.show()

def plot_corr_history(h, title,plot_name=None,checkpoint=0):
    
    print("Plotting correlation history...")
    plot_name_corr=plot_name+"_corr.png"
    corr_plot = plt.figure()
    corr_plot.plot(h.history['p_corr'][5:], label = "Train cor", color = "blue")
    corr_plot.plot(h.history['val_p_corr'][5:], label = "Validation cor", color = "red")
    corr_plot.xlabel('Epochs')
    corr_plot.title(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name and checkpoint == 0:
        #plt.legend()
        corr_plot.savefig(plot_name_corr)
        corr_plot.close()
    else:
        pass

class TrainingHistoryPlotter:
    """
    A utility class for tracking and plotting training metrics in PyTorch.
    """
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, metrics):
        """
        Update history with metrics from current epoch/iteration.
        
        Args:
            metrics (dict): Dictionary of metric names and values
        """
        for metric_name, value in metrics.items():
            # Convert tensor to float if needed
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.history[metric_name].append(value)
    
    def plot(self, figsize=(12, 8), save_path=None):
        """
        Plot all tracked metrics.
        
        Args:
            figsize (tuple): Figure size (width, height)
            save_path (str, optional): Path to save the plot
        """
        n_metrics = len(self.history)
        
        # Determine grid layout
        n_cols = min(2, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(self.history.items()):
            ax = axes[i]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, 'o-', label=metric_name)
            ax.set_title(f'{metric_name} Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True)
            
            # Add moving average for noisy metrics
            if len(values) > 5:
                window_size = max(3, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                moving_avg_epochs = range(window_size, len(values) + 1)
                ax.plot(moving_avg_epochs, moving_avg, 'r-', label=f'{metric_name} (Moving Avg)')
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
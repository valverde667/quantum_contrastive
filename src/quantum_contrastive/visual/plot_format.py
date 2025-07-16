import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def set_plot_style():
    sns.set_palette("deep")
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_style("ticks")

    mpl.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 100,
            "figure.figsize": (8, 6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],  # Clean and available everywhere
        }
    )

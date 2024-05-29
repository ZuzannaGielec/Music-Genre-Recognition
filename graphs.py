import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("../features_30sec.xlsx", sheet_name="Sheet1", header = 0)

feature_cols = ["tempo", "rms_mean", "rms_var", "zero_crossing_mean", "zero_crossing_var", "bandwidth_mean",
                "bandwidth_var", "rolloff_mean", "rolloff_var", "harmonic_mean", "harmonic_var", "stft_mean", 
                "stft_var"]
X = data[feature_cols]
Y = data.label

def plotter(feature, X, Y):
    plt.scatter(X[feature], Y)
    plt.title(feature)
    plt.savefig(feature + ".png")
    plt.clf()

for col in feature_cols:
    plotter(col, X, Y)

"""
c = ['xkcd:purple' if y=="blues" else 
    'xkcd:blue' if y=="country" else
    'xkcd:green' if y=="classical" else
    'xkcd:pink' if y=="disco" else
    'xkcd:red' if y=="pop" else
    'xkcd:orange' if y=="reggae" else
    'xkcd:light blue' if y=="rock" else
    'xkcd:yellow' if y=="hiphop" else
    'xkcd:spruce' if y=="jazz" else
    'xkcd:lilac'    for y in Y]
plt.scatter(X.stft_var, X.rms_mean, c=c)
plt.show()
"""
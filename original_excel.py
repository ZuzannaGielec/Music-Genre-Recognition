import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz as gv

data = pd.read_csv("../Data/features_30_sec.csv")

feature_cols = ["chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var", "spectral_centroid_mean", "spectral_centroid_var", 
                "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", 
                "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", 
                "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", 
                "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", "mfcc10_mean", "mfcc10_var",
                "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean",
                "mfcc15_var", "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var",	
                "mfcc20_mean", "mfcc20_var"]

X = data[feature_cols]
Y = data.label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7)
dt = dt.fit(X_train, Y_train)

rf = RandomForestClassifier(n_estimators=500, max_depth = 8)
rf = rf.fit(X_train, Y_train)

gb = GradientBoostingClassifier(n_estimators=100)
gb = gb.fit(X_train, Y_train)

dt_pred = dt.predict(X_test)
dt_pred2 = dt.predict(X_train)
rf_pred = rf.predict(X_test)
rf_pred2 = rf.predict(X_train)
gb_pred = rf.predict(X_test)
gb_pred2 = rf.predict(X_train)

print("Decision Tree Validation Accuracy:",metrics.accuracy_score(Y_test, dt_pred))
print("Decision Tree Train Accuracy:",metrics.accuracy_score(Y_train, dt_pred2))
print("Random Forest Validation Accuracy:",metrics.accuracy_score(Y_test, rf_pred))
print("Random Forest Train Accuracy:",metrics.accuracy_score(Y_train, rf_pred2))
print("Gradient Boost Validation Accuracy:",metrics.accuracy_score(Y_test, gb_pred))
print("Gradient Boost Train Accuracy:",metrics.accuracy_score(Y_train, gb_pred2))
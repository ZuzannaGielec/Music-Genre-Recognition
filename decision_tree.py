import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz as gv

data = pd.read_excel("../features_30sec.xlsx", "Sheet1", header = 0)

feature_cols = ["tempo", "rms_mean", "rms_var", "zero_crossing_mean", "zero_crossing_var", "bandwidth_mean",
                "bandwidth_var", "rolloff_mean", "rolloff_var", "harmonic_mean", "harmonic_var", "stft_mean", 
                "stft_var"]

X = data[feature_cols]
Y = data.label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7)

clf = clf.fit(X_train, Y_train)

dot_data = export_graphviz(clf, out_file = None, 
                                feature_names = feature_cols,
                                class_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                                filled = True)

# Draw graph
graph = gv.Source(dot_data, format="png") 
graph.render("decision_tree_graphivz")

Y_pred = clf.predict(X_test)
Y_pred2 = clf.predict(X_train)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Train Accuracy:",metrics.accuracy_score(Y_train, Y_pred2))
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from dtreeviz.trees import dtreeviz

house = load_boston()
X = house.data
y = house.target

reg = RandomForestRegressor(n_estimators=100,max_depth=3,max_features='auto',min_samples_leaf=4,bootstrap=True,n_jobs=-1,random_state=0)

reg.fit(X,y)

viz = dtreeviz(reg.estimators_[-1],X,y,feature_names=house.feature_names,title="last tree - House data")

viz.save("decision_tree_house.svg")

from sklearn import preprocessing
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

scaler = preprocessing.StandardScaler()
#scaler.fit(X)
xsa = scaler.fit_transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
xs.sample(5)
#print(xs.sample(5))

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
#print(gmm)
gmm.fit(xs)
y_gmm = gmm.predict(xs)
print(y_gmm)

colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_gmm], s=40)
plt.title('GMM Classification')

print('Accuracy Score = ', accuracy_score(y, y_gmm))

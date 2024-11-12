import sklearn
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

path = '/Users/willpelech/Desktop/Movie_Rec_SVM/action.csv'

df = pd.read_csv(path, index_col='movie_id')

x = df[['rating','director_id']].values # Select the first two features for visualization purposes


y_encoder = LabelEncoder()
x_encoder = LabelEncoder()


y_enc = y_encoder.fit_transform(df[['genre']].values.ravel())
type_enc = x_encoder.fit_transform(df['rating'])
country_enc = x_encoder.fit_transform(df['director_id'])

x_enc = np.column_stack((type_enc, country_enc))

X_train, X_test, Y_train, Y_test = train_test_split(x_enc,y_enc, test_size=0.2 )

svm = SVC(kernel='poly',degree=5,random_state=42)
svm.fit(X_train,Y_train)




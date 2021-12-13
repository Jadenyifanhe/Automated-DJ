import os
import pickle

from sklearn.externals import joblib

theme_pca = joblib.load(os.path.join('song_theme_pca_model_2.pkl'))
print(theme_pca)

with open('song_theme_pca_model_pickle.pkl', 'wb') as f:
    pickle.dump(theme_pca, f)

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

# データセットの読み込み
data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
    )

pixel_values, targets = data
targets = targets.astype(int)

single_image = pixel_values.iloc[1, :].values.reshape(28, 28)

plt.imshow(single_image, cmap='gray')

# t-SNE
tsne = manifold.TSNE(n_components=2, random_state=42)

transformed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])

# データフレームの作成
tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=['x', 'y', 'targets']
    )

tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)

# 可視化
grid = sns.FacetGrid(tsne_df, hue='targets', size=8)
grid.map(plt.scatter, 'x', 'y').add_legend()
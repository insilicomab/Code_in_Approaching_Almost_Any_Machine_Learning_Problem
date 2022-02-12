# ライブラリのインポート
import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':
    
    df = pd.read_csv('train.csv')
    
    # kfoldという新しい列を作り、-1で初期化
    df['skfold'] = -1
    
    '''
    サンプルをシャッフル
    引数frac=1とすると、すべての行数分のランダムサンプリングをすることになり、
    全体をランダムに並び替える（シャッフルする）ことに等しい。
    '''
    df = df.sample(frac=1).reset_index(drop=True)
    
    # 目的変数を取り出す
    y = df['y']
    
    # KFoldクラスの初期化
    skf = model_selection.StratifiedKFold(n_splits=5)
    
    # kfold列を埋める
    for fold, (train_, val_) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_, 'skfold'] = fold
    
    # データセットを新しい列と共に保存
    df.to_csv('train_skfolds.csv', index=False)
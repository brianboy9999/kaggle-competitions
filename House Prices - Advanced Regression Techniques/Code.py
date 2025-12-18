# ===============================
# 0. 套件匯入
# ===============================
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# ===============================
# 1. 讀取資料
# ===============================
train_data = pd.read_csv('train.csv')

# ===============================
# 2. 自訂前處理 Transformer
# ===============================
class HousePricesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.none_categorical = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageQual', 'GarageFinish', 'GarageType', 'GarageCond',
            'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual',
            'BsmtFinType1', 'MasVnrType'
        ]

    def fit(self, X, y = None):
        X = X.copy()

        # LotFrontage
        self.lotfrontage_median_ = (
            X.groupby('Neighborhood')['LotFrontage'].median()
        )

        # Electrical
        self.electrical_mode_ = X['Electrical'].mode()[0]

        return self

    def transform(self, X):
        X = X.copy()

        # 1.NaN為沒有
        for col in self.none_categorical:
            X[col] = X[col].fillna('None')

        # 2.數值型缺失
        X['LotFrontage'] = X.apply(
            lambda r: self.lotfrontage_median_.get(
                r['Neighborhood'], self.lotfrontage_median_.median()
            ) if pd.isnull(r['LotFrontage']) else r['LotFrontage'],
            axis = 1
        )

        X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)

        # 3.少量缺失
        X['Electrical'] = X['Electrical'].fillna(self.electrical_mode_)

        return X
    
# ===============================
# 3. 切資料
# ===============================
X = train_data.drop('SalePrice', axis = 1)
y = train_data['SalePrice']

X_train, X_val, y_train, y_test = train_test_split(
    X, y , test_size = 0.2, random_state = 42
)

# ===============================
# 4. Pipeline
# ===============================
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('cleaner', HousePricesPreprocessor()),
    ('encoder', preprocess),
    ('model', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)

test_data = pd.read_csv('test.csv')

test_preds = pipeline.predict(test_data)

submission = pd.DataFrame({
    'Id': test_data['Id'],          # House Prices 用的是 Id
    'SalePrice': test_preds
})

submission.to_csv('submission.csv', index=False)
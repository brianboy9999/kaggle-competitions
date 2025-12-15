# ===============================
# 0. 套件匯入
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# 1. 讀取資料
# ===============================
train_data = pd.read_csv('train.csv')

# ===============================
# 2. 自訂前處理 Transformer
# ===============================
RARE_TITLES = [
    'Dr', 'Rev', 'Col', 'Major', 'Don', 'Lady',
    'Sir', 'Capt', 'Countess', 'Jonkheer', 'Dona'
]

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()

        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X['Title'] = X['Title'].replace(RARE_TITLES, 'Rare')

        self.title_age_median_ = X.groupby('Title')['Age'].median()
        self.fare_median_ = X['Fare'].median()
        self.embarked_mode_ = X['Embarked'].mode()[0]

        return self

    def transform(self, X):
        X = X.copy()

        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X['Title'] = X['Title'].replace('Mme', 'Mrs')
        X['Title'] = X['Title'].replace(RARE_TITLES, 'Rare')

        X['Age'] = X.apply(
            lambda r: self.title_age_median_.get(
                r['Title'], self.title_age_median_.median()
            ) if pd.isnull(r['Age']) else r['Age'],
            axis=1
        )

        X['Fare'] = X['Fare'].fillna(self.fare_median_)
        X['Fare'] = np.log1p(X['Fare'])
        
        X['Embarked'] = X['Embarked'].fillna(self.embarked_mode_)

        X['TicketPrefix'] = X['Ticket'].str.extract('([A-Za-z]+)', expand=False)
        X['TicketPrefix'] = X['TicketPrefix'].fillna('None')

        X['HasCabin'] = X['Cabin'].notnull().astype(int)
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

        return X

# ===============================
# 3. 切資料
# ===============================
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ===============================
# 4. Pipeline
# ===============================
categorical_cols = ['Sex', 'Embarked', 'Title', 'TicketPrefix']
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'HasCabin', 'FamilySize', 'IsAlone']

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('cleaner', TitanicPreprocessor()),
    ('drop_cols', FunctionTransformer(
        lambda df: df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    )),
    ('encoder', preprocess),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# ===============================
# 5. 訓練模型
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# 6. 模型評估
# ===============================
y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

# ===============================
# 7. 預測 test.csv & 產生 submission
# ===============================
test_data = pd.read_csv('test.csv')

test_preds = pipeline.predict(test_data)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_preds
})

submission.to_csv('submission.csv', index=False)
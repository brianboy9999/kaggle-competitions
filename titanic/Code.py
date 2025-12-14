# 1.讀取/檢視資料 
import pandas as pd

train_data = pd.read_csv('train.csv')

# 2.清洗資料
from sklearn.model_selection import train_test_split

# One-Hot Encoding
# Title
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'],'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Don', 'Lady', 'Sir', 'Capt', 'Countess', 'Jonkheer'], 'Rare')
# Ticket
train_data['TicketPrefix'] = train_data['Ticket'].str.extract('([A-Za-z]+)', expand = False)
train_data['TicketPrefix'] = train_data['TicketPrefix'].fillna('None', inplace = False)

# 補缺失值
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Age'] = train_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
train_data['HasCabin'] = train_data['Cabin'].notnull().astype(int)

# 3. 訓練集/測試集
X = train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
Y = train_data['Survived']
categorial_cols = ['Sex', 'Embarked', 'Title', 'TicketPrefix']
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'HasCabin']
X_train, X_val, Y_train, Y_val =  train_test_split(X, Y, test_size = 0.1, random_state = 42)

# 4.定義步驟
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

process = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorial_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

pipeline = Pipeline(steps = [
    ('proprocessor', process),
    ('classifier', RandomForestClassifier(random_state = 42))
])

# 5.訓練模型
pipeline.fit(X_train, Y_train)

# 6.模型評估
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 預測驗證集
Y_pred = pipeline.predict(X_val)

# 評估準確率(模型整體預測正確的比例)
acc = accuracy_score(Y_val, Y_pred)

# 混淆矩陣(在哪一類預測表現好或差)
cm = confusion_matrix(Y_val, Y_pred)

# 分類報告(Precision(預測存活中，有多少是真的存活), Recall(實際存活的人中，有多少被模型正確預測), F1-score(Precision 和 Recall 的調和平均))
class_report = classification_report(Y_val, Y_pred)

# 7.輸出結果
test_data = pd.read_csv('test.csv')

# 清洗資料
# One-Hot Encoding
# Title
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
test_data['Title'] = test_data['Title'].replace('Ms','Miss')
test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Dona'], 'Rare')
# Ticket
test_data['TicketPrefix'] = test_data['Ticket'].str.extract('([A-Za-z]+)', expand = False)
test_data['TicketPrefix'] = test_data['TicketPrefix'].fillna('None', inplace = False)


# 補缺失值
train_title_age_median = train_data.groupby('Title')['Age'].median()
test_data['Age'] = test_data.apply(
    lambda row: train_title_age_median[row['Title']] if pd.isnull(row['Age']) else row['Age'],
    axis=1
)
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())
test_data['HasCabin'] = test_data['Cabin'].notnull().astype(int)

# # 預測
X_test = test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
preds = pipeline.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': preds
})
submission.to_csv('submission.csv', index = False)
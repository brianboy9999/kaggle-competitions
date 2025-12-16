資料集特徵:
1.分類預測
2.標準為Accuracy

用 sklearn Pipeline，把「只從 train 學到的清洗規則」
安全地套用到 validation / test，避免任何資料洩漏(test.csv 從頭到尾沒參與 fit，所有補值都來自 train，Pipeline 保證順序正確)。

特殊資料清洗
1.Name取姓氏分類
2.Fare取中位數補值再取Log(分布太廣)
3.Embarked取中位數補值
4.Ticket測試是否有前綴詞
5.Cabin One-Hot Encoding
6.新增FamilySize 為家庭數量
7.新增IsAlone為是否一個人

為什麼要用 class，不用 function？
1.記住 train 的統計量
2.train / test 同一套規則
3.能進 Pipeline

BaseEstimator:
1.可以被 Pipeline 使用
2.可以被 clone
3.可以被 GridSearchCV 管理參數

TransformerMixin:
1.保證有 fit_transform
2.Pipeline 知道什麼時候要 transform

fit() vs transform():
fit():
只看 train
記住 median / mode
不真的改資料
transform():
套用規則
train / val / test 都走這裡
不重新算統計量
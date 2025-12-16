# ===============================
# 0. 套件匯入
# ===============================
import pandas as pd

# ===============================
# 1. 讀取資料
# ===============================
train_data = pd.read_csv('train.csv')
missing = train_data.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])

# ===============================
# 2. 自訂前處理 Transformer
# ===============================

資料集特徵: 1.特徵非常多 2.缺失值很多 3.回歸預測 4.標準為RMSE

缺失值分三類: 
1.NaN為沒有-特徵出現率極低 
2.數值型缺失-用統計量補
2.1. LotFrontage鄰接寬度(必為資料缺失)
同一個Neighborhood的房子，地塊大小、臨街寬度、價格分布都比較相近，依Neighborhood分組取LotFrontage中位數
2.2. GarageYrBlt磚石外牆面積(沒外牆即NaN)
2.3. MasVnrArea車庫建造年份(沒車庫即NaN)
3.少量缺失-mode(類型) / median(數值)
3.1 Electrical 電氣系統

__init__:
會不會影響模型行為的可調性，如果我換一種做法，模型行為就會改變的東西
Titanic的rare_titles只影響單一欄位，不太可能修改，於Class不一定需要_init_
House Prices的none_categorical牽涉多欄位，可能須隨時驗調整，於Class可放_init_

select_dtype(indclude, exclude)(from pandas):
用於選擇符合特定數據類型的 DataFrame 列

相較於Titanic:
1.資料、特徵較多
2.大量類別特徵

脊回歸(Rigid Regression):
1.為L2 正則化，正則化用以減少過擬合之誤差
2.脊迴歸不會將迴歸係數歸零，因此不執行特徵選擇
3.脊迴歸無法在嚴重多重共線性下區分預測因子效應
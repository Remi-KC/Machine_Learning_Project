import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# set path
path1 = "/Users/remikc/Programming/Python/ML_Project/Data/"
path2 = "/Users/remikc/Programming/Python/ML_Project/Plot/"

# load data
netflix = pd.read_csv(path1+"netflix.csv")

# 中文設定
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["font.size"] = 13

#%%
netflix.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5337 entries, 0 to 5336
Data columns (total 5 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   type           5337 non-null   object 
 1   listed_in      5164 non-null   object 
 2   country_main   4941 non-null   object 
 3   best_director  5337 non-null   int64  
 4   averageRating  5337 non-null   float64
dtypes: float64(1), int64(1), object(3)
memory usage: 208.6+ KB
'''

# 遺漏值處理：填入新標籤Unknown，避免損失太多資料
netflix = netflix.fillna("Unknown").reset_index(drop=True)

#%% one-hot encoding

# 類別欄位轉成 one-hot encoding: type, country_main, best_director(因為yes/no無順序關係) 
ohe = OneHotEncoder(sparse=False, dtype=int)
ohe_data = pd.DataFrame(ohe.fit_transform(netflix[["type", "country_main", "best_director"]]), 
                        columns=list(ohe.categories_[0])+list(ohe.categories_[1])+["no_best_director", "best_director"])

# listed_in 欄位轉成 one-hot encoding (MultiLabelBinarizer)
netflix["listed_in"] = netflix["listed_in"].str.split(", ")
mlb = MultiLabelBinarizer()
genre = pd.DataFrame(mlb.fit_transform(netflix["listed_in"]), columns=mlb.classes_)

# 合併資料
data = pd.concat([ohe_data, genre], axis=1)
# 將Unknown特徵刪除(原本為遺漏值)，避免混淆結果
data.drop(columns=["Unknown"], inplace=True)

#%% 建模

X = data
y = netflix["averageRating"]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=10)
lm = LinearRegression().fit(XTrain, yTrain)
pred_test = lm.predict(XTest)

#%% 績效評分 --> 差強人意
'''
可能有其他應該納入的特徵，也可能現有資料集裡面沒有真的很關鍵的特徵

可以考慮探索其他特徵（本資料集或者另找其他資料）
本資料集中尚未探索的特徵為：「演員」和「影片描述」
'''
print("MSE: %.2f" % np.mean((yTest-pred_test)**2))
print("R-squared: %.2f" % lm.score(XTest, yTest))
# MSE: 1.03
# R-squared: 0.26

#%% 計算相關值
res = pearsonr(np.array(yTest), pred_test)
print(res)# (0.5172091626329249, 4.111238613058888e-74)
print("r = %.2f, p = %.4f" % (res[0], res[1]))
# r = 0.52, p = 0.0000

#%% 作圖：預測評分 vs 實際評分 對照圖
predict = pd.DataFrame({"yTest":np.array(yTest), "pred":pred_test})
g = sns.lmplot(x="yTest", y="pred", data=predict, aspect=2)

# 繪製刪除線
for ax in g.axes.flat:
    d = .9  
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0, 0], [0.02, 0.04], transform=ax.transAxes, **kwargs)
    ax.plot([0.01, 0.02], [0, 0], transform=ax.transAxes, **kwargs)

plt.title("Netflix 影片評分｜ 模型預測結果", loc="left")
plt.title("r = .52| p < .001", loc="right")
plt.xlabel("實際：影片評分")
plt.ylabel("預\n測\n：\n影\n片\n評\n分", rotation=0, labelpad=12)
plt.xticks(np.arange(2,10,0.5))
plt.yticks(np.arange(5,9,0.5), [""]+list(np.arange(5.5,9,0.5)))
plt.grid()

# 存檔
plt.savefig(path2+"corr.png", bbox_inches="tight", dpi=200)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set path
path1 = "/Users/remikc/Programming/Python/ML_Project/Data/"
path2 = "/Users/remikc/Programming/Python/ML_Project/Plot/"

# load data
netflix = pd.read_csv(path1+"netflix_new.csv")

# 中文設定
plt.rcParams["font.family"] = ["Heiti TC"]
plt.rcParams["font.size"] = 13

#%% 
# 資料概覽
netflix.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8807 entries, 0 to 8806
Data columns (total 24 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   platform        8807 non-null   object 
 1   show_id         8807 non-null   object 
 2   type            8807 non-null   object 
 3   title           8807 non-null   object 
 4   director        6173 non-null   object 
 5   cast            7982 non-null   object 
 6   country         7976 non-null   object 
 7   country_main    7976 non-null   object 
 8   date_added      8797 non-null   object 
 9   release_year    8807 non-null   int64  
 10  rating          8717 non-null   object 
 11  duration        8804 non-null   object 
 12  listed_in       8734 non-null   object 
 13  description     8807 non-null   object 
 14  tconst          8614 non-null   object 
 15  averageRating   8330 non-null   float64
 16  numVotes        8330 non-null   float64
 17  f_date_add      8797 non-null   object 
 18  f_date_add_Ym   8797 non-null   object 
 19  f_date_add_m    8797 non-null   float64
 20  f_year_release  8807 non-null   object 
 21  diff_days       8797 non-null   float64
 22  diff_months     8797 non-null   float64
 23  diff_years      8797 non-null   float64
dtypes: float64(6), int64(1), object(17)
memory usage: 1.6+ MB

'''
#%% 影視分類 vs 影片評分(numVotes>1000)
showtype = netflix[netflix["numVotes"]>1000][["type", "averageRating"]]
# 按照 影視分類 計算平均評分
showtype_avg = showtype.groupby(["type"]).mean().reset_index()
showtype_avg.sort_values(by=["averageRating"], ascending=False, inplace=True) 

showtype_score = showtype.merge(showtype_avg, how="left", on=["type"])
showtype_score.sort_values(by=["averageRating_y"], ascending=False, inplace=True) 

#%% 作圖：影視分類 vs 影片評分(numVotes>1000)

plt.figure(figsize = (8, 8), dpi=200)

sns.barplot(x="type", y="averageRating", data=showtype_avg,
            alpha=.8, palette="rocket")
sns.stripplot(x="type", y="averageRating_x", data=showtype_score,
              edgecolor='brown', linewidth=0.2, jitter=0.3, palette="rocket")

plt.title("Netflix 影視分類 vs 影片評分", fontsize=21, loc="left")
plt.title("評分數>1000人", fontsize=15, loc="right")
plt.xlabel("")
plt.xticks(np.arange(2),["電視節目", "電影節目"])
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"type.png", bbox_inches="tight")
plt.show()


#%% 國家 vs 影片評分(numVotes>1000)
loc = netflix[netflix["numVotes"]>1000][["country_main", "averageRating"]]

# 確認空值比例會不會太高
print("country_main 空值比例 = %.2f%%" % (loc["country_main"].isna().sum()/len(loc)*100))
print("averageRating 空值比例 = %.2f%%" % (loc["averageRating"].isna().sum()/len(loc)*100))
# country_main 空值比例 = 4.98%
# averageRating 空值比例 = 0.00%

# 空值比例不高，決定刪除
loc.dropna(inplace=True)

# 按照 國家 計算平均評分
loc_avg = loc.groupby(["country_main"]).mean().reset_index()

# 確認國家影片數
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(loc["country_main"].value_counts())
'''
United States           2366
India                    555
United Kingdom           427
Japan                    182
Canada                   162
France                   146
Spain                    132
South Korea              125
Turkey                    98
Australia                 70
Germany                   68
Mexico                    61
China                     55
Italy                     46
Egypt                     44
Hong Kong                 41
Brazil                    35
Argentina                 34
Thailand                  27
Indonesia                 25
Poland                    24
Ireland                   23
Taiwan                    23
Denmark                   20
Norway                    18
Israel                    16
Nigeria                   15
Sweden                    15
Philippines               15
Belgium                   14
Netherlands               13
South Africa              13
Colombia                  12
New Zealand               11
Russia                    10
Pakistan                   9
Singapore                  9
Chile                      8
Switzerland                7
Austria                    7
Romania                    7
Lebanon                    7
Iceland                    6
United Arab Emirates       6
Finland                    6
Saudi Arabia               6
Bulgaria                   5
Portugal                   4
Hungary                    4
Czech Republic             4
Vietnam                    3
Croatia                    2
Malaysia                   2
Uruguay                    2
Georgia                    2
Ukraine                    2
Jordan                     2
Peru                       2
Cambodia                   2
Serbia                     2
Kenya                      2
Bangladesh                 2
Mauritius                  2
West Germany               1
Slovenia                   1
Iran                       1
Luxembourg                 1
Greece                     1
Puerto Rico                1
Mozambique                 1
Somalia                    1
Name: country_main, dtype: int64
'''    
# 只保留>=10部影片的國家 
loc_ = loc.groupby(["country_main"]).size().reset_index().rename(columns={0:"count"})
loc_ = loc_[loc_["count"]>=10]

loc_avg = loc_.merge(loc_avg, how="left", on=["country_main"]) 
loc_avg.sort_values(by=["averageRating"], ascending=False, inplace=True) 

loc_score = loc_.merge(loc, how="left", on=["country_main"]).merge(loc_avg, how="left", on=["country_main"])
loc_score.sort_values(by=["averageRating_y"], ascending=False, inplace=True) 

# 各國評分有些微差異
loc_avg.describe()
'''
             count  averageRating
count    23.000000      23.000000
mean    207.347826       6.685826
std     488.512455       0.306537
min      23.000000       6.066667
25%      34.500000       6.501598
50%      61.000000       6.625714
75%     139.000000       6.804882
max    2366.000000       7.417033
'''
#%% 作圖：國家 vs 影片評分(numVotes>1000)

plt.figure(figsize = (25, 6), dpi=200)

sns.barplot(x="country_main", y="averageRating", data=loc_avg, alpha=.8)
sns.stripplot(x="country_main", y="averageRating_x", data=loc_score, 
              edgecolor='brown', linewidth=0.2, jitter=0.3)

plt.title("Netflix 各國影片評分", fontsize=21, loc="left")
plt.title("評分數>1000人| 評分數>1000人的作品>=10部", fontsize=15, loc="right")
plt.xlabel("")
plt.xticks(rotation=75)
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"country.png", bbox_inches="tight")
plt.show()


#%% 影片分類處理
# Netflix Main Genre Categories 官網大分類項
# https://www.whats-on-netflix.com/news/the-netflix-id-bible-every-category-on-netflix/ 
'''
Action & Adventure (1365)
Anime (7424)
Children & Family (783)
Classic (31574)
Comedies (6548)
Documentaries (6839)
Dramas (5763)
Horror (8711)
Music (1701)
Romantic (8883)
Sci-fi & Fantasy (1492)
Sports (4370)
Thrillers (8933)
TV Shows (83)
'''

# 確認原始資料各分類標籤數量，根據官網大分類項標示要刪除跟合併的項目
netflix["listed_in"].str.split(", ").explode().value_counts()
'''
International                   4103 X
Dramas                          3190
Comedies                        2255
Action & Adventure              1027
Romantic                         986
Documentaries                    869
Independent                      756 X
Children & Family                641 
Thrillers                        634
Crime                            470 X
Kids                             451 Children & Family 
Horror                           432
Stand-Up Comedy & Talk Shows     399 Comedies
Docuseries                       395 Documentaries 
Music & Musicals                 375 Music
Sci-Fi & Fantasy                 327 
Reality                          255 X
British                          253 X
Anime                            247
Sports                           219
Classic & Cult                   215 Classic
Spanish-Language                 174 X
Korean                           151 X
LGBTQ                            102 X
Mysteries                         98 X
Science & Nature                  92 X
Teen                              69 X
Faith & Spirituality              65 X
Name: listed_in, dtype: int64
'''

#%% Modify genres to reduce categories
# 只保留Netflix官網公告的大分類項目
new_genre = {
    "listed_in":{
        r'International|Independent|Crime|Reality|British|Spanish-Language|Korean|LGBTQ|Mysteries|Science & Nature|Teen|Faith & Spirituality':"",
        r'Kids':"Children & Family",
        r'Stand-Up Comedy & Talk Shows':"Comedies",
        r'Docuseries':"Documentaries",
        r'Music & Musicals':"Music",
        r'Classic & Cult':"Classic"}
    }
netflix = netflix.replace(new_genre, regex=True)

# 清除除非字元開頭和結尾，以及多餘的逗號
netflix = netflix.replace({"listed_in":{r'^\W+|\W+$':"",
                                        r'(, )+':", "}}, regex=True)
# 沒有值的地方補NAN
netflix["listed_in"] = netflix["listed_in"].replace("", np.nan)

# 確認空值比例會不會太高
print("有分類標籤的筆數佔總資料%.2f%%" % (netflix["listed_in"].count()/len(netflix)*100))
# 有分類標籤的筆數佔總資料95.70%

# 計算每個標籤總數
print(netflix["listed_in"].str.split(", ").explode().value_counts())
'''
Dramas                3190
Comedies              2654
Documentaries         1264
Children & Family     1092
Action & Adventure    1027
Romantic               986
Thrillers              634
Horror                 432
Music                  375
Sci-Fi & Fantasy       327
Anime                  247
Sports                 219
Classic                215
Name: listed_in, dtype: int64
'''

#%% 平均評分 vs 影視類型*影片分類(only numVotes>1000)

# 取出需要的欄位
genre = netflix[netflix["numVotes"]>1000][["type", "listed_in", "averageRating"]]

# 將分類欄位改成list形式
genre["listed_in"] = genre["listed_in"].str.split(", ")

# 展開分類欄位，使每一列只有一個分類
genre_ex = genre.explode("listed_in", ignore_index=True)

genre_ex.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8262 entries, 0 to 8261
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   type           8262 non-null   object 
 1   listed_in      8089 non-null   object 
 2   averageRating  8262 non-null   float64
dtypes: float64(1), object(2)
memory usage: 193.8+ KB
'''
# 計算空值比例
print("listed_in 空值比例 = %.2f%%" % (genre_ex["listed_in"].isna().sum()/len(genre_ex)*100))
print("averageRating 空值比例 = %.2f%%" % (genre_ex["averageRating"].isna().sum()/len(genre_ex)*100))
# listed_in 空值比例 = 2.09%
# averageRating 空值比例 = 0.00%

# 空值比例不高，決定直接刪除
genre_ex.dropna(inplace=True)

# 按照 影視類型/影片分類 計算平均評分
genre_avg = genre_ex.groupby(["type", "listed_in"]).mean().reset_index()
genre_avg.sort_values(by=["type", "averageRating"], ascending=[True, False], inplace=True)

genre_score = genre_ex.merge(genre_avg, how="left", on=["type", "listed_in"])
genre_score.sort_values(by=["type", "averageRating_y"], ascending=[True, False], inplace=True)

#%% 作圖：平均評分 vs 影視類型*影片分類 
'''
Insights:
1. 電視的評分較電影來得高
2. 不同類型的影片評分有差異
3. 在電視和電影中，受到好評的影片類型不完全相同       
'''
plt.figure(figsize = (18, 15), dpi=200)
plt.subplots_adjust(hspace=0.5)


plt.subplot(2,1,1)
sns.barplot(x="listed_in", y="averageRating", data=genre_avg[genre_avg["type"]=="TV Show"], alpha=.8)
sns.stripplot(x="listed_in", y="averageRating_x", data=genre_score[genre_score["type"]=="TV Show"], 
              edgecolor='brown', linewidth=0.2, jitter=0.3)

plt.title("Netflix電視｜各個影片類型評分", fontsize=21, loc="left")
plt.title("評分人數>1000", fontsize=15, loc="right")
plt.xlabel("")
plt.xticks(rotation=75)
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
              
plt.subplot(2,1,2)
sns.barplot(x="listed_in", y="averageRating", data=genre_avg[genre_avg["type"]=="Movie"], alpha=.8)
sns.stripplot(x="listed_in", y="averageRating_x", data=genre_score[genre_score["type"]=="Movie"], 
              edgecolor='brown', linewidth=0.2, jitter=0.3)              

plt.title("Netflix電影｜各個影片類型評分", fontsize=21, loc="left")
plt.title("評分人數>1000", fontsize=15, loc="right")
plt.xlabel("")
plt.xticks(rotation=75)
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
sns.despine()

# 存檔
plt.savefig(path2+"type_genre.png", bbox_inches="tight")
plt.show()

#%% 發行年份vs評分 corr
year = netflix[["release_year", "averageRating", "numVotes"]]
year.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8807 entries, 0 to 8806
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   release_year   8807 non-null   int64  
 1   averageRating  8330 non-null   float64
 2   numVotes       8330 non-null   float64
dtypes: float64(2), int64(1)
memory usage: 206.5 KB
'''
year.dropna(inplace=True)

#%% 作圖：發行年份vs評分 corr 
# 1990年以前發行的影片較集中在高評分區，其餘的影片評分分佈較廣，但整體而言和評分沒有相關
sns.lmplot(x="release_year", y="averageRating", 
           data=year[year["numVotes"]>1000], aspect=1.5)

plt.title("影片發行年份 vs 評分", fontsize=14, loc="left")
plt.title("評分數>1000人", fontsize=10, loc="right")
plt.xlabel("發行年份", fontsize=12, labelpad=5)
plt.ylabel("影\n片\n評\n分", rotation=0, fontsize=12, labelpad=8, loc="center")
plt.yticks(np.arange(11))
plt.xticks(np.arange(1940,2022,5), rotation=75)
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"release.png", bbox_inches="tight", dpi=200)
plt.show()


#%% 時間差vs評分 corr
diff = netflix[["diff_days", "averageRating", "numVotes"]]
diff.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8807 entries, 0 to 8806
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   diff_days      8797 non-null   float64
 1   averageRating  8330 non-null   float64
 2   numVotes       8330 non-null   float64
dtypes: float64(3)
memory usage: 206.5 KB
'''
diff.dropna(inplace=True)

#%% 作圖：時間差vs評分 corr 
# 時間差>11000天的影片較集中在高評分區，其餘的影片評分分佈較廣，但整體而言和評分沒有相關
sns.lmplot(x="diff_days", y="averageRating", 
           data=diff[(diff["numVotes"]>1000)&(diff["diff_days"]>=0)],
           aspect=1.5)

plt.title("(上架-發布)時間差 vs 影片評分", fontsize=14, loc="left")
plt.title("評分數>1000人", fontsize=10, loc="right")
plt.xlabel("時間差(日)", fontsize=12, labelpad=5)
plt.ylabel("影\n片\n評\n分", rotation=0, fontsize=12, labelpad=8, loc="center")
plt.xticks(np.arange(0,29000,1000), rotation=75)
plt.yticks(np.arange(11))
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"diff.png", bbox_inches="tight", dpi=200)
plt.show()

#%% 時間差>11000天的影片 和 1990年以前發行的影片 只有24部重疊，決定還是視為兩個不同特徵
len(netflix[(netflix["numVotes"]>1000)&(netflix["diff_days"]>=11000)])
# 177
len(netflix[(netflix["numVotes"]>1000)&(netflix["release_year"]<=1990)]) 
# 163
len(set(netflix[(netflix["numVotes"]>1000)&(netflix["diff_days"]>=11000)])&set(netflix[(netflix["numVotes"]>1000)&(netflix["release_year"]<=1990)]))
# 24

#%% rating vs averageRating

rating = netflix[["rating", "averageRating", "numVotes"]]
rating.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8807 entries, 0 to 8806
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   rating         8717 non-null   object 
 1   averageRating  8330 non-null   float64
 2   numVotes       8330 non-null   float64
dtypes: float64(2), object(1)
memory usage: 206.5+ KB
'''
rating.dropna(inplace=True)
rating = rating[rating["numVotes"]>1000]

rating_avg = rating.groupby(["rating"]).mean().reset_index()
rating_avg.sort_values(by=["averageRating"], ascending=False, inplace=True)

rating_score = rating.merge(rating_avg, how="left", on=["rating"])
rating_score.sort_values(by=["averageRating_y"], ascending=False, inplace=True)

#%% 作圖：rating vs averageRating --> 差異微乎其微
plt.figure(figsize = (10, 6), dpi=200)

sns.barplot(x="rating", y="averageRating", data=rating_avg, alpha=.8, palette="rocket")
sns.stripplot(x="rating", y="averageRating_x", data=rating_score, 
              edgecolor='brown', linewidth=0.2, jitter=0.3, palette="rocket")

plt.title("Netflix 分級影片評分", fontsize=21, loc="left")
plt.title("評分數>1000人", fontsize=15, loc="right")
plt.xlabel("影片分級", fontsize=18, labelpad=10)
plt.xticks(np.arange(4),["輔導級", "限制級", "保護級", "普遍級"])
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"rating.png", bbox_inches="tight")
plt.show()

#%% 爬蟲:最佳導演名單 https://parade.com/1048720/maramovies/best-directors/
import requests
from bs4 import BeautifulSoup 
import re

url = "https://parade.com/1048720/maramovies/best-directors/"
res = requests.get(url)

# 爬蟲取得最佳導演名單
best_directors = []
if res.status_code == 200:
    bs = BeautifulSoup(res.text, "lxml")
    for h2 in bs.find_all("h2"):
        if re.search(r'\d+\.+', h2.text) is not None:
            name = re.search(r'\d+.+', h2.text).group(0)
            best_directors.append(re.search(r'[a-zA-Z]+[\.|\-|\s|a-zA-Z]+[a-zA-Z]', name).group(0))
else:
    print("取得網頁內容失敗")

#%% 最佳導演(是/否) director --> (1/0)

# 只保留第一位導演資料
netflix["main_director"] = netflix["director"].apply(lambda x: str(x).split(", ")[0])

# 新增判斷導演是否為最佳導演的欄位
netflix["best_director"] = 0
# 如果main_director是最佳導演，就將best_director的值改為1
netflix["best_director"] = netflix["main_director"].apply(lambda x: 1 if x in best_directors else 0)

#%% 有無最佳導演 vs 評分
directors = netflix[["best_director", "averageRating", "numVotes"]]

directors_avg = directors[directors["numVotes"]>1000].groupby(["best_director"]).mean().reset_index()

directors_score = directors.merge(directors_avg, how="left", on=["best_director"])

#%% 作圖：有無最佳導演 vs 評分 --> 有著名導演的影片評分較高一些，且分布較集中在高分區
plt.figure(figsize = (8, 8), dpi=200)

sns.barplot(x="best_director", y="averageRating", data=directors_avg,
            alpha=.8, order=[1,0], palette="rocket")
sns.stripplot(x="best_director", y="averageRating_x", data=directors_score, order=[1,0],
              edgecolor='brown', linewidth=0.2, jitter=0.3, palette="rocket")

plt.title("Netflix 有無著名導演 vs 影片評分", fontsize=21, loc="left")
plt.title("評分數>1000人", fontsize=15, loc="right")
plt.xlabel("")
plt.xticks(np.arange(2),["有（著名導演）", "無（著名導演）"])
plt.ylabel("影\n片\n評\n分\n:\n平\n均\n數", rotation=0, fontsize=18, labelpad=15, loc="center")
plt.yticks(np.arange(11))
plt.grid()
              
sns.despine()

# 存檔
plt.savefig(path2+"director.png", bbox_inches="tight")
plt.show()

#%% EDA小結
'''
* 對影片評分可能有影響的特徵：
    1.影視類型(電影/電視)
    2.影片類型(紀錄片、經典、動畫、劇情...)
    3.出品國家
    4.有無著名導演
    
* 對影片評分沒有影響的特徵：
    1.影片發行年份
    2.影片上架-影片發行)時間差
    3.影片分級
'''

#%% 清理資料  --> 儲存

# 將>1000人評分之影片數過少(<10部)的國家設定為np.nan，避免國別預測不準確
delete_loc = list(loc["country_main"].value_counts().index[35:])
'''
['Pakistan', 'Singapore', 'Chile', 'Switzerland', 'Austria', 'Romania',
       'Lebanon', 'Iceland', 'United Arab Emirates', 'Finland', 'Saudi Arabia',
       'Bulgaria', 'Portugal', 'Hungary', 'Czech Republic', 'Vietnam',
       'Croatia', 'Malaysia', 'Uruguay', 'Georgia', 'Ukraine', 'Jordan',
       'Peru', 'Cambodia', 'Serbia', 'Kenya', 'Bangladesh', 'Mauritius',
       'West Germany', 'Slovenia', 'Iran', 'Luxembourg', 'Greece',
       'Puerto Rico', 'Mozambique', 'Somalia']
'''
netflix["country_main"] = netflix["country_main"].apply(lambda x: np.nan if x in delete_loc else x) 

# 只保留評分人次>1000的資料，否則評分本身沒有可信度
netflix = netflix[netflix["numVotes"]>1000]


# 只保留需要的欄位
netflix = netflix[["type","listed_in", "country_main", "best_director", "averageRating"]]
netflix.to_csv(path1+"netflix.csv", index=False)



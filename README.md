# Netflix 影片評分預測模型 ---Machine Learning Project

>完成於2022.08.31<br>
>花費時間：約3個晚上
<br>

## 一、	專題目標
延續[前一份專題](https://github.com/Remi-KC/DataAnalysisProject_with_WebCrawling)，就現有資料集，探索哪些特徵可能影響影片評分，建構評分預測模型。<br>
<br>
 
## 二、	資料概述
### Netflix平台影片與評分資料（8807筆）：
* [來源](https://github.com/Remi-KC/DataAnalysisProject_with_WebCrawling "看資料來源")：使用前一份專題清理、合併過後的Netflix資料。 
* 欄位：片名、影視分類、影片類型、國家、發行年份、上架日期、影片分級、（上架-發行）時間差、導演、演員、作品描述、影片評分、影片評分人次
<br>

## 三、	資料清理（清理後共5337筆）
>程式碼請參考：[1_Cleansing_and_EDA.py](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/1_Cleansing_and_EDA.py "看我的程式碼")
*	資料篩選：只保留評分人次>1000的資料，否則評分本身可信度不足。
*	影片類型：合併細項，只保留官網公告的大分類項目，將分類數降低到13種。
*	國家欄位：影片數量不足10部的國家，更改其國家為空值。因為筆數太少的情況下，這些國別與評分的關係有可能只是湊巧。
*	導演欄位：爬蟲75位最佳導演名單，比對影片是否由這些導演執導，新增類別變項best_director（1=由最佳導演執導，0=非由最佳導演執導）。
<br>

## 四、	資料分析
>程式碼請參考：[1_Cleansing_and_EDA.py](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/1_Cleansing_and_EDA.py "看我的程式碼")
### 1. 國家 vs 影片評分
![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/country.png)
* 各國影片評分有差異。作品最受好評的前三名依序是：日本、紐西蘭、英國，臺灣則名列第12。
* 美國雖然是影片數量最多的國家，但是他的影片評分分佈也最廣，從低分到高分的影片數量都很多，並沒有特別集中在哪一個分數區間。
<br>

### 2. 影視類別x影片類型 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/type_genre.png" width="844.571"><br>
* 電視的評分看起來都比電影來得高，這部分待後續確認。
* 不同影片類型的評分有差異，電影和電視的前三名都包含：經典、動畫、紀錄片。
* 在電視和電影節目中，受到好評的前三名雷同，但後續的排名不完全相同。影視類別和影片類型可能相互影響評分，後續建模有必要個別納入這兩個特徵。
<br>

### 3. 影視類別 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/type.png" width="500"><br>
* 電視作品的評分的確高於電影作品。而且這個差距將近一個標準差(std=1.148)。
<br>

### 4. 有無著名導演執導 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/director.png" width="500"><br>
* 有著名導演執導的作品評分比較高，且分佈多數集中在6分以上的高分區。但這類作品數量相對很少。
* 沒有著名導演執導的作品評分比較低，評分幾乎涵蓋所有區間。
<br>

### 5. 影片分級 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/rating.png" width="572"><br>
* 不同分級的影片評分幾乎沒有差異，且評分的分佈情形也雷同。
<br>

### 7. 影片發行年份 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/release.png" width="582.4"><br>
* 整體而言，發行年份與影片評分無關。
* 就分佈情形來看，1990年以前發行的作品，評分相較更集中在高分區。但這類作品很少。
<br>

### 8. (上架-發行)時間差 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/diff.png" width="582.4"><br>
* 整體而言，「作品從上架到發行的時間差」不影響影片評分。
* 就分佈情形來看，時間差大於11000天（約30年）的作品，評分相對更集中在高分區。但這類作品很少。
<br>


### 小結
* 對影片評分可能有影響的特徵：
    * 1.影視類別（電影/電視）
    * 2.影片類型（紀錄片、經典、動畫、劇情......）
    * 3.出品國家
    * 4.有無著名導演
    
* 對影片評分沒有影響的特徵：
    * 1.影片發行年份
    * 2.（影片上架-影片發行）時間差
    * 3.影片分級
<br>

## 五、	迴歸模型
>程式碼請參考：[2_ML_Regression.py](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/2_ML_Regression.py "看我的程式碼")
### 1. 遺漏值處理：
* 有遺漏值的欄位：國家、影片類型
* 填入"Unknown"，避免損失太多資料

### 2. 特徵處理：
* 類別資料轉成one-hot encoding：影視類別、影片類型、國家、有無著名導演
* 由於影片類型中，部分欄位有複數個值，使用MultiLabelBinarizer轉成one-hot encoding
* 類別特徵全部轉換完後，將Unknown特徵刪除(原本為遺漏值)，避免混淆結果。
 
### 3. 模型結果與結論：
![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/corr.png)<br>
* 實際評分和預測評分有中度相關，且相關達顯著(r = .52；p < 0.001)。
* 模型可解釋比例約26%，MSE為1.03。尚有探索其他特徵的空間。
<br>

![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/coef.png)<br>
* 可以看到，現有資料集中，大部分特徵的預測力都偏低，因此模型結果差強人意。
* 迴歸係數絕對值>0.5的為部分國家(波蘭/紐西蘭/南韓)，以及影片類型(經典流行/恐怖片/紀錄片)
<br>

![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/1.png)<br>
* 刪除預測力過低(coef < 0.1)的特徵，模型解釋力仍然維持0.26。
* 刪除 coef > 0.1 的特徵值，會降低模型解釋力。
* 可見，以目前的資料量，解釋力最高就是0.26。
* 如果要提高解釋力，應該增加其他特徵。
<br>

* 後續方向：
  * 依據迴歸係數選擇特徵
  * 調整處理「導演」特徵的方式，改為該片導演總執導次數。
  * 「演員」特徵比照「導演」處理，計算前三位演員(主演)出演的總數。
  * 探索文字特徵：「影片描述」
  * 另找其他資料





# Netflix 影片評分預測模型 ---Machine Learning Project
<br>

## 一、	專題目標：
延續前一份專題，就現有資料集，探索哪些特徵可能影響影片評分，建構評分預測模型。
<br>
 
## 二、	資料概述：
### Netflix平台影片與評分資料（8807筆）：
* 來源：使用前一份專題清理、合併過後的Netflix資料。
* 欄位：片名、影視分類、影片類型、國家、發行年份、上架日期、影片分級、（上架-發行）時間差、導演、演員、作品描述、影片評分、影片評分人次
<br>

## 三、	資料清理：
*	資料篩選：只保留評分人次>1000的資料，否則評分本身可信度不足。
*	影片類型：合併細項，只保留官網公告的大分類項目，將分類數降低到13種。
*	國家欄位：影片數量不足10部的國家，更改其國家為空值。因為這些國別與評分的關係不可信。
*	導演欄位：爬蟲75位最佳導演名單，比對影片是否由這些導演執導，新增類別變項best_director（1=由最佳導演執導，0=非由最佳導演執導）。
<br>

## 四、	資料分析：
### 1. 國家 vs 影片評分
![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/country.png)

### 2. 影視類別x影片類型 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/type_genre.png" width="844.571" height="780.285">

### 3. 影視類別 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/type.png" width="500" height="500">

### 4. 有無著名導演執導 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/director.png" width="500" height="500">

### 5. 影片分級 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/rating.png" width="572" height="379.33">

### 7. 影片發行年份 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/release.png" width="582.4" height="419.6">

### 8. (上架-發行)時間差 vs 影片評分
<img src="https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/diff.png" width="582.4" height="430">


## 五、	迴歸模型：
![image](https://github.com/Remi-KC/Machine_Learning_Project/blob/main/Plot/corr.png)




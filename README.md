# 2016-PHM-Data-Challenge
Machine learning analysis for predicting CMP average material removal rate using the 2016 PHM Data Challenge dataset.

### 分析目標

- 建立 wafer-stage 層級的製程特徵
- 比較 XGBoost、Random Forest 與 SVR 的預測表現
- 評估集成模型的效果
- 分析重要特徵，提出製程與設備管理建議

### 資料來源

本專案使用 PHM Data Challenge 2016 CMP 資料

資料取得方式、欄位與檔案配置請參考
[data/README.md](data/README.md)

預測目標為 `AVG_REMOVAL_RATE`，
樣本以 `WAFER_ID` 與 `STAGE` 識別

### 分析流程

1. 資料讀取與品質檢查
2. 資料清理與時序特徵工程
3. 探索性資料分析
4. 特徵篩選、模型調參與交叉驗證
5. 模型比較與集成。
6. 特徵解釋與製程建議

### 專案結構

- `CMP.ipynb`：完整分析流程與結果說明
- `src/`：資料處理、模型訓練與評估函式
- `data/`：資料說明、原始資料與處理後資料
- `results/`：圖表與結果表格

## 執行方式

### 1. 安裝套件

```bash
python -m pip install -r requirements.txt
```

### 2. 準備資料

依照 [資料說明](data/README.md)，
將資料放入 `data/raw/` 的對應資料夾

### 3. 執行分析

從專案根目錄啟動 Jupyter：

```bash
jupyter notebook
```

開啟 `CMP.ipynb`，選擇安裝好套件的 Python 環境，由上往下執行所有 cells

## 分析結果

待整理後重新執行，再補上模型比較表與代表性圖表

## 分析限制

待補上資料排除規則、評估範圍與方法限制

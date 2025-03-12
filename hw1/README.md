# 糖尿病二分类模型训练与评估

本项目的目标是训练一个二分类模型，用于预测糖尿病。代码使用了多种机器学习算法（Gradient Boosting、Random Forest 和 XGBoost），并通过网格搜索进行超参数调优，最终比较各模型的表现。

## 1. 代码结构

### 1.1 数据加载 (`Data Loading`)
- 从指定路径加载糖尿病数据集 (`diabetes.csv`)。

```python
file_path = "E:/diabetes.csv"
df = pd.read_csv(file_path)
```

### 1.2 数据清洗 (`Data Cleaning`)
- 将无效的零值替换为 `NaN`（如 `Glucose`、`BloodPressure` 等列）。
- 使用 `StandardScaler` 对数据进行标准化。
- 使用 `KNNImputer` 对缺失值进行插补。

```python
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Outcome'])), columns=df.columns[:-1])

imputer = KNNImputer(n_neighbors=3, weights="distance")
df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns[:-1])
df_imputed["Outcome"] = df["Outcome"].astype(int)
```

### 1.3 特征工程 (`Feature Engineering`)
- 使用 `SelectKBest` 和 `f_classif` 进行特征选择，选择最相关的 6 个特征。

```python
selector = SelectKBest(f_classif, k=6)
X_selected = selector.fit_transform(X, y)
selected_cols = df.columns[:-1][selector.get_support()]
```

### 1.4 处理类别不平衡 (`Handle Class Imbalance`)
- 使用 `SMOTE` 对少数类样本进行过采样，解决类别不平衡问题。

```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)
```

### 1.5 模型训练与调优 (`Model Training & Tuning`)
- 将数据集划分为训练集和测试集。
- 使用 `GridSearchCV` 对模型进行超参数调优，优化召回率 (`Recall`)。
- 支持 `Gradient Boosting`、`Random Forest` 和 `XGBoost` 三种模型。

```python
def train_and_evaluate_model(model, param_grid, model_name):
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='recall'
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    print(f"\nModel: {model_name}")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Recall Score: {recall_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Confusion Matrix (Test Set) - {model_name}")
    plt.show()
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        plt.figure(figsize=(10, 4))
        sns.barplot(x=importance, y=selected_cols, palette="viridis")
        plt.title(f"Feature Importance Ranking - {model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()
    
    return best_model, recall_score(y_test, y_pred)
```

### 1.6 结果比较 (`Compare Model Results`)
- 输出各模型的召回率，并进行比较。

```python
print("\nModel Recall Scores:")
print(f"Gradient Boosting: {gb_recall:.4f}")
print(f"Random Forest: {rf_recall:.4f}")
print(f"XGBoost: {xgb_recall:.4f}")
```

## 2. 模型参数网格

### 2.1 Gradient Boosting
```python
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
```

### 2.2 Random Forest
```python
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5]
}
```

### 2.3 XGBoost
```python
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
```

## 3. 输出结果
- **最佳参数**：每个模型在网格搜索中找到的最佳超参数。
- **召回率**：模型在测试集上的召回率。
- **分类报告**：包括精确率、召回率、F1 分数等指标。
- **混淆矩阵**：可视化模型在测试集上的分类结果。
- **特征重要性**：展示每个特征对模型预测的贡献程度。

## 4. 运行环境
- Python 3.x
- 依赖库：`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `xgboost`



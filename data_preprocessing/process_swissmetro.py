import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('../data/raw/swissmetro.csv')
data = data[data['CHOICE'] != 0]
print("数据基本信息:")
print(data.info())

columns_to_drop = ['SP', 'ID', 'GROUP', 'SURVEY', 'TRAIN_AV', 'CAR_AV', 'SM_AV']
data = data.drop(columns=columns_to_drop, errors='ignore')

print(f"\n删除后的数据列名数量: {len(data.columns)}")
print("剩余的列名:")
print(data.columns.tolist())

target = 'CHOICE'

X = data.drop(columns=[target], axis=1)
y = data[target]
print(y.max())
print(y.min())

numeric_features = [
    'TRAIN_TT', 'TRAIN_CO', 'TRAIN_HE',
    'SM_TT', 'SM_CO', 'SM_HE',
    'CAR_TT', 'CAR_CO'
]

categorical_features = [
    'PURPOSE', 'FIRST', 'TICKET', 'WHO',
    'LUGGAGE', 'AGE', 'MALE', 'INCOME',
    'GA', 'ORIGIN', 'DEST', 'SM_SEATS'
]


missing_numeric = [col for col in numeric_features if col not in X.columns]
missing_categorical = [col for col in categorical_features if col not in X.columns]

if missing_numeric:
    print(f"警告: 以下数值型变量不存在于数据集中: {missing_numeric}")
if missing_categorical:
    print(f"警告: 以下分类变量不存在于数据集中: {missing_categorical}")


numeric_features = [col for col in numeric_features if col in X.columns]
categorical_features = [col for col in categorical_features if col in X.columns]

print(f"\n数值型特征数量: {len(numeric_features)}")
print(f"分类特征数量: {len(categorical_features)}")


for col in categorical_features:
    X[col] = X[col].astype(str)


mapping = {}
current_code = 0

for feature in categorical_features:
    temp = X[feature].astype(int)
    unique_values = sorted(temp.unique())
    for value in unique_values:
        key = f"{feature}_{value}"
        if key not in mapping:
            mapping[key] = current_code
            current_code += 1

print("\n类别特征编码映射:")
for key, value in mapping.items():
    print(f"{key}: {value}")

for feature in categorical_features:
    X[feature] = X[feature].apply(lambda x: int(mapping[f"{feature}_{x}"]))

total_unique_categorical = len(mapping)
print(f"\n总的唯一分类特征值个数: {total_unique_categorical}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = preprocessing_pipeline.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42, stratify=y
)
print("\n预处理后的训练集和测试集形状:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
print("\n训练集的类别分布:")
print(y_train.value_counts())
print("\n测试集的类别分布:")
print(y_test.value_counts())

num_features = numeric_features
ordinal_feature_names = categorical_features
all_feature_names = list(num_features) + list(ordinal_feature_names)

print(f"\n特征名称数量: {len(all_feature_names)}, 预处理后的特征数量: {X_processed.shape[1]}")
if X_processed.shape[1] != len(all_feature_names):
    print(f"警告: 预处理后的特征数量 ({X_processed.shape[1]}) 与特征名称数量 ({len(all_feature_names)}) 不匹配。")
    print("请检查预处理步骤和特征列表。")
else:
    print("\n所有特征名称及其属性类型:")
    for feature in all_feature_names:
        print(feature)

X_train_df = pd.DataFrame(X_train, columns=all_feature_names)
X_test_df = pd.DataFrame(X_test, columns=all_feature_names)

y_train_df = y_train.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)

# # 保存为CSV文件
# X_train_df.to_csv('../data/processed/swissmetro/X_train.csv', index=False)
# X_test_df.to_csv('../data/processed/swissmetro/X_test.csv', index=False)
# y_train_df.to_csv('../data/processed/swissmetro/y_train.csv', index=False)
# y_test_df.to_csv('../data/processed/swissmetro/y_test.csv', index=False)
#
# print("\n预处理后的数据已保存为 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv'")

# ============================================================
# BANK MARKETING PREDICTION PROJECT
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 2. DATA LOADING
# ============================================================

file_path = "bank_marketing.csv"

# Automatically detects comma or semicolon separated CSV
df = pd.read_csv(
    file_path,
    sep=None,
    engine="python"
)

print("=" * 70)
print("DATA LOADED SUCCESSFULLY")
print("=" * 70)

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 4. DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

df.info()

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_data = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

print(missing_data)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# ============================================================
# 7. FIND TARGET COLUMN
# ============================================================

# Bank Marketing datasets generally use "y".
# Some versions use "deposit".

if "y" in df.columns:

    target_column = "y"

elif "deposit" in df.columns:

    target_column = "deposit"

else:

    raise ValueError(
        "Target column not found. "
        "Expected 'y' or 'deposit'."
    )


print("\nTarget column:", target_column)


# ============================================================
# 8. CONVERT TARGET TO 0 AND 1
# ============================================================

print("\nOriginal target values:")

print(
    df[target_column].value_counts()
)


if df[target_column].dtype == "object":

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df[target_column] = (
        df[target_column]
        .map({
            "yes": 1,
            "no": 0
        })
    )


elif set(
    df[target_column].dropna().unique()
).issubset({0, 1}):

    df[target_column] = df[target_column].astype(int)


else:

    raise ValueError(
        "Target variable must contain yes/no or 0/1."
    )


# Remove rows where target could not be converted
df = df.dropna(
    subset=[target_column]
)

df[target_column] = (
    df[target_column].astype(int)
)


print("\nTarget after conversion:")

print(
    df[target_column].value_counts()
)


print("\nTarget percentage:")

print(
    (
        df[target_column]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# 9. EDA - TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x=target_column
)

plt.title("Target Variable Distribution")

plt.xlabel(
    "Subscription (0 = No, 1 = Yes)"
)

plt.ylabel("Count")

plt.tight_layout()

plt.show()


# ============================================================
# 10. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

X_temp = df.drop(
    columns=[target_column]
)

numeric_columns = (
    X_temp
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_columns = (
    X_temp
    .select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)


print("\nNumerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 11. NUMERICAL EDA
# ============================================================

for column in numeric_columns:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(
        "Distribution of " + column
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. CATEGORICAL EDA
# ============================================================

for column in categorical_columns:

    # Limit very high-cardinality columns
    if df[column].nunique() <= 15:

        plt.figure(figsize=(8, 4))

        sns.countplot(
            data=df,
            x=column,
            hue=target_column
        )

        plt.title(
            column + " vs Target"
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()


# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

plt.figure(figsize=(12, 8))

correlation_matrix = df.corr(
    numeric_only=True
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 14. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# AGE GROUP
if "age" in df.columns:

    df["age_group"] = pd.cut(
        df["age"],
        bins=[
            0,
            25,
            35,
            45,
            55,
            65,
            np.inf
        ],
        labels=[
            "Young",
            "Adult",
            "Middle_Age",
            "Senior",
            "Older",
            "Very_Old"
        ]
    )


# BALANCE GROUP
if "balance" in df.columns:

    df["balance_group"] = pd.cut(
        df["balance"],
        bins=[
            -np.inf,
            0,
            1000,
            5000,
            10000,
            np.inf
        ],
        labels=[
            "Negative",
            "Low",
            "Medium",
            "High",
            "Very_High"
        ]
    )


# CAMPAIGN GROUP
if "campaign" in df.columns:

    df["campaign_group"] = pd.cut(
        df["campaign"],
        bins=[
            0,
            1,
            2,
            5,
            np.inf
        ],
        labels=[
            "1",
            "2",
            "3_5",
            "6_plus"
        ]
    )


# PREVIOUS CONTACT
if "previous" in df.columns:

    df["previous_contact"] = np.where(
        df["previous"] > 0,
        "Yes",
        "No"
    )


print("Feature Engineering Completed.")

print("\nNew columns:")

print(
    [
        column
        for column in [
            "age_group",
            "balance_group",
            "campaign_group",
            "previous_contact"
        ]
        if column in df.columns
    ]
)


# ============================================================
# 15. TRAIN TEST SPLIT
# ============================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)


# ============================================================
# 16. PREPARE DATA FOR WOE
# ============================================================

woe_train = X_train.copy()
woe_test = X_test.copy()


# Convert category columns to object
for column in woe_train.columns:

    if str(
        woe_train[column].dtype
    ) == "category":

        woe_train[column] = (
            woe_train[column]
            .astype(object)
        )

        woe_test[column] = (
            woe_test[column]
            .astype(object)
        )


# ============================================================
# 17. WOE FUNCTION
# ============================================================

def calculate_woe_iv(
    data,
    feature,
    target
):

    temp = data[
        [feature, target]
    ].copy()

    temp[feature] = (
        temp[feature]
        .astype(str)
    )

    grouped = (
        temp
        .groupby(
            feature,
            dropna=False
        )[target]
        .agg(
            ["count", "sum"]
        )
    )

    grouped["bad"] = (
        grouped["sum"]
    )

    grouped["good"] = (
        grouped["count"]
        - grouped["sum"]
    )

    total_good = (
        grouped["good"].sum()
    )

    total_bad = (
        grouped["bad"].sum()
    )

    grouped["dist_good"] = (
        grouped["good"]
        / total_good
    )

    grouped["dist_bad"] = (
        grouped["bad"]
        / total_bad
    )

    # Avoid division by zero
    grouped["dist_good"] = (
        grouped["dist_good"]
        .replace(0, 0.0001)
    )

    grouped["dist_bad"] = (
        grouped["dist_bad"]
        .replace(0, 0.0001)
    )

    # WOE
    grouped["WOE"] = np.log(
        grouped["dist_good"]
        / grouped["dist_bad"]
    )

    # IV
    grouped["IV"] = (
        (
            grouped["dist_good"]
            - grouped["dist_bad"]
        )
        * grouped["WOE"]
    )

    total_iv = (
        grouped["IV"].sum()
    )

    return grouped, total_iv


# ============================================================
# 18. CREATE NUMERICAL BINS
# ============================================================

numeric_features = (
    woe_train
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_features = (
    woe_train
    .select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)

woe_features = []


for feature in numeric_features:

    try:

        _, bins = pd.qcut(
            woe_train[feature],
            q=5,
            duplicates="drop",
            retbins=True
        )

        bins[0] = -np.inf
        bins[-1] = np.inf

        train_bin_column = (
            feature + "_bin"
        )

        test_bin_column = (
            feature + "_bin"
        )

        woe_train[
            train_bin_column
        ] = pd.cut(
            woe_train[feature],
            bins=bins,
            include_lowest=True
        ).astype(str)

        woe_test[
            test_bin_column
        ] = pd.cut(
            woe_test[feature],
            bins=bins,
            include_lowest=True
        ).astype(str)

        woe_features.append(
            train_bin_column
        )

    except Exception:

        print(
            "Could not create bins for:",
            feature
        )


# Add categorical features
woe_features.extend(
    categorical_features
)


# ============================================================
# 19. CALCULATE WOE AND IV
# ============================================================

print("\n" + "=" * 70)
print("WOE AND IV ANALYSIS")
print("=" * 70)


train_for_woe = pd.concat(
    [
        woe_train,
        y_train.rename("target")
    ],
    axis=1
)


iv_results = []

woe_tables = {}


for feature in woe_features:

    table, iv = calculate_woe_iv(
        train_for_woe,
        feature,
        "target"
    )

    woe_tables[feature] = table

    iv_results.append(
        {
            "Feature": feature,
            "IV": iv
        }
    )


iv_df = pd.DataFrame(
    iv_results
)


iv_df = iv_df.sort_values(
    by="IV",
    ascending=False
)


print("\nIV Results:")

print(
    iv_df.to_string(index=False)
)


# ============================================================
# 20. SELECT FEATURES USING IV
# ============================================================

selected_features = (
    iv_df[
        iv_df["IV"] >= 0.02
    ]["Feature"]
    .tolist()
)


# If no feature passes 0.02,
# select top 10 features

if len(selected_features) == 0:

    selected_features = (
        iv_df
        .head(10)["Feature"]
        .tolist()
    )


print("\nSelected Features:")

print(selected_features)


# ============================================================
# 21. CREATE WOE MAPPING
# ============================================================

woe_mapping = {}


for feature in selected_features:

    woe_mapping[feature] = (
        woe_tables[feature]["WOE"]
        .to_dict()
    )


# ============================================================
# 22. APPLY WOE TRANSFORMATION
# ============================================================

def apply_woe(
    data,
    features,
    mappings
):

    result = pd.DataFrame(
        index=data.index
    )

    for feature in features:

        result[feature] = (
            data[feature]
            .astype(str)
            .map(
                mappings[feature]
            )
        )

        # Unknown values
        # receive WOE = 0
        result[feature] = (
            result[feature]
            .fillna(0)
        )

    return result


X_train_woe = apply_woe(
    woe_train,
    selected_features,
    woe_mapping
)

X_test_woe = apply_woe(
    woe_test,
    selected_features,
    woe_mapping
)


X_train_woe = (
    X_train_woe
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

X_test_woe = (
    X_test_woe
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


print("\nWOE transformed data:")

print(
    X_train_woe.head()
)


# ============================================================
# 23. VIF ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("VIF ANALYSIS")
print("=" * 70)


# Remove constant columns
vif_columns = []

for column in X_train_woe.columns:

    if (
        X_train_woe[column]
        .nunique()
        > 1
    ):

        vif_columns.append(
            column
        )


X_vif = X_train_woe[
    vif_columns
].copy()


vif_results = []


for i, column in enumerate(
    X_vif.columns
):

    try:

        vif_value = (
            variance_inflation_factor(
                X_vif.values,
                i
            )
        )

    except Exception:

        vif_value = np.inf

    vif_results.append(
        {
            "Feature": column,
            "VIF": vif_value
        }
    )


vif_df = pd.DataFrame(
    vif_results
)


vif_df = vif_df.sort_values(
    by="VIF",
    ascending=False
)


print(
    vif_df.to_string(
        index=False
    )
)


# ============================================================
# 24. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)


scaler = StandardScaler()


X_train_lr = (
    scaler.fit_transform(
        X_train_woe
    )
)


X_test_lr = (
    scaler.transform(
        X_test_woe
    )
)


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


logistic_model.fit(
    X_train_lr,
    y_train
)


y_pred_lr = (
    logistic_model.predict(
        X_test_lr
    )
)


y_prob_lr = (
    logistic_model
    .predict_proba(
        X_test_lr
    )[:, 1]
)


# ============================================================
# 25. LOGISTIC REGRESSION RESULTS
# ============================================================

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

lr_precision = precision_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_auc = roc_auc_score(
    y_test,
    y_prob_lr
)


print(
    "Accuracy:",
    round(lr_accuracy, 4)
)

print(
    "Precision:",
    round(lr_precision, 4)
)

print(
    "Recall:",
    round(lr_recall, 4)
)

print(
    "F1 Score:",
    round(lr_f1, 4)
)

print(
    "ROC-AUC:",
    round(lr_auc, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_lr,
        zero_division=0
    )
)


# ============================================================
# 26. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================================
# 27. DECISION TREE DATA
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE")
print("=" * 70)


X_tree = df.drop(
    columns=[target_column]
)

y_tree = df[target_column]


X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree,
    y_tree,
    test_size=0.20,
    random_state=42,
    stratify=y_tree
)


# ============================================================
# 28. DECISION TREE PREPROCESSING
# ============================================================

tree_numeric = (
    X_train_tree
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


tree_categorical = (
    X_train_tree
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)


tree_preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            SimpleImputer(
                strategy="median"
            ),
            tree_numeric
        ),

        (
            "categorical",
            Pipeline(
                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            tree_categorical
        )
    ]
)


# ============================================================
# 29. DECISION TREE MODEL
# ============================================================

decision_tree = Pipeline(
    steps=[

        (
            "preprocessor",
            tree_preprocessor
        ),

        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 30. TRAIN DECISION TREE
# ============================================================

decision_tree.fit(
    X_train_tree,
    y_train_tree
)


y_pred_dt = (
    decision_tree.predict(
        X_test_tree
    )
)


y_prob_dt = (
    decision_tree
    .predict_proba(
        X_test_tree
    )[:, 1]
)


# ============================================================
# 31. DECISION TREE RESULTS
# ============================================================

dt_accuracy = accuracy_score(
    y_test_tree,
    y_pred_dt
)

dt_precision = precision_score(
    y_test_tree,
    y_pred_dt,
    zero_division=0
)

dt_recall = recall_score(
    y_test_tree,
    y_pred_dt,
    zero_division=0
)

dt_f1 = f1_score(
    y_test_tree,
    y_pred_dt,
    zero_division=0
)

dt_auc = roc_auc_score(
    y_test_tree,
    y_prob_dt
)


print(
    "Accuracy:",
    round(dt_accuracy, 4)
)

print(
    "Precision:",
    round(dt_precision, 4)
)

print(
    "Recall:",
    round(dt_recall, 4)
)

print(
    "F1 Score:",
    round(dt_f1, 4)
)

print(
    "ROC-AUC:",
    round(dt_auc, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_test_tree,
        y_pred_dt,
        zero_division=0
    )
)


# ============================================================
# 32. DECISION TREE CONFUSION MATRIX
# ============================================================

cm_dt = confusion_matrix(
    y_test_tree,
    y_pred_dt
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================================
# 33. DECISION TREE PLOT
# ============================================================

print("\nPlotting Decision Tree...")


# Get fitted preprocessing object
fitted_preprocessor = (
    decision_tree
    .named_steps[
        "preprocessor"
    ]
)


# Get fitted decision tree
fitted_tree = (
    decision_tree
    .named_steps[
        "classifier"
    ]
)


# Get feature names after encoding
feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# Clean feature names
feature_names = [
    name
    .replace(
        "numeric__",
        ""
    )
    .replace(
        "categorical__",
        ""
    )
    for name in feature_names
]


plt.figure(
    figsize=(25, 15)
)


plot_tree(
    fitted_tree,
    feature_names=feature_names,
    class_names=[
        "No",
        "Yes"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)


plt.title(
    "Decision Tree - Bank Marketing",
    fontsize=18
)

plt.tight_layout()


# Save the tree plot
plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 34. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)


param_grid = {

    "classifier__max_depth": [
        3,
        4,
        5,
        6,
        8
    ],

    "classifier__min_samples_split": [
        2,
        5,
        10
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        5
    ],

    "classifier__criterion": [
        "gini",
        "entropy"
    ]
}


grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train_tree,
    y_train_tree
)


# ============================================================
# 35. BEST PARAMETERS
# ============================================================

print("\nBest Parameters:")

print(
    grid_search.best_params_
)


print(
    "\nBest Cross Validation ROC-AUC:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 36. BEST / TUNED DECISION TREE
# ============================================================

best_decision_tree = (
    grid_search.best_estimator_
)


y_pred_best = (
    best_decision_tree.predict(
        X_test_tree
    )
)


y_prob_best = (
    best_decision_tree
    .predict_proba(
        X_test_tree
    )[:, 1]
)


# ============================================================
# 37. TUNED TREE RESULTS
# ============================================================

best_accuracy = accuracy_score(
    y_test_tree,
    y_pred_best
)

best_precision = precision_score(
    y_test_tree,
    y_pred_best,
    zero_division=0
)

best_recall = recall_score(
    y_test_tree,
    y_pred_best,
    zero_division=0
)

best_f1 = f1_score(
    y_test_tree,
    y_pred_best,
    zero_division=0
)

best_auc = roc_auc_score(
    y_test_tree,
    y_prob_best
)


print("\n" + "=" * 70)
print("TUNED DECISION TREE RESULTS")
print("=" * 70)


print(
    "Accuracy:",
    round(best_accuracy, 4)
)

print(
    "Precision:",
    round(best_precision, 4)
)

print(
    "Recall:",
    round(best_recall, 4)
)

print(
    "F1 Score:",
    round(best_f1, 4)
)

print(
    "ROC-AUC:",
    round(best_auc, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y_test_tree,
        y_pred_best,
        zero_division=0
    )
)


# ============================================================
# 38. TUNED DECISION TREE PLOT
# ============================================================

print("\nPlotting Tuned Decision Tree...")


best_preprocessor = (
    best_decision_tree
    .named_steps[
        "preprocessor"
    ]
)


best_tree = (
    best_decision_tree
    .named_steps[
        "classifier"
    ]
)


best_feature_names = (
    best_preprocessor
    .get_feature_names_out()
)


best_feature_names = [
    name
    .replace(
        "numeric__",
        ""
    )
    .replace(
        "categorical__",
        ""
    )
    for name in best_feature_names
]


plt.figure(
    figsize=(25, 15)
)


plot_tree(
    best_tree,
    feature_names=best_feature_names,
    class_names=[
        "No",
        "Yes"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)


plt.title(
    "Tuned Decision Tree - Bank Marketing",
    fontsize=18
)


plt.tight_layout()


# Save tuned tree
plt.savefig(
    "tuned_decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 39. TUNED TREE CONFUSION MATRIX
# ============================================================

cm_best = confusion_matrix(
    y_test_tree,
    y_pred_best
)


plt.figure(figsize=(6, 5))


sns.heatmap(
    cm_best,
    annot=True,
    fmt="d",
    cmap="Oranges"
)


plt.title(
    "Tuned Decision Tree Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()


# ============================================================
# 40. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Tuned Decision Tree"
    ],

    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        best_accuracy
    ],

    "Precision": [
        lr_precision,
        dt_precision,
        best_precision
    ],

    "Recall": [
        lr_recall,
        dt_recall,
        best_recall
    ],

    "F1 Score": [
        lr_f1,
        dt_f1,
        best_f1
    ],

    "ROC-AUC": [
        lr_auc,
        dt_auc,
        best_auc
    ]
})


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)


print(
    results.round(4)
    .to_string(index=False)
)


# ============================================================
# 41. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(12, 6))


results_plot = results.set_index(
    "Model"
)


results_plot[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].plot(
    kind="bar",
    figsize=(12, 6)
)


plt.title(
    "Model Performance Comparison"
)

plt.ylabel("Score")

plt.xlabel("Model")

plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()


# ============================================================
# 42. ROC CURVE COMPARISON
# ============================================================

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr
)


fpr_dt, tpr_dt, _ = roc_curve(
    y_test_tree,
    y_prob_dt
)


fpr_best, tpr_best, _ = roc_curve(
    y_test_tree,
    y_prob_best
)


plt.figure(figsize=(8, 6))


plt.plot(
    fpr_lr,
    tpr_lr,
    label=(
        "Logistic Regression "
        "AUC = "
        + str(round(lr_auc, 3))
    )
)


plt.plot(
    fpr_dt,
    tpr_dt,
    label=(
        "Decision Tree "
        "AUC = "
        + str(round(dt_auc, 3))
    )
)


plt.plot(
    fpr_best,
    tpr_best,
    label=(
        "Tuned Decision Tree "
        "AUC = "
        + str(round(best_auc, 3))
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve Comparison"
)


plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 43. BEST MODEL
# ============================================================

best_model_index = (
    results["ROC-AUC"].idxmax()
)


best_model = (
    results.loc[
        best_model_index
    ]
)


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    "Model:",
    best_model["Model"]
)


print(
    "Accuracy:",
    round(
        best_model["Accuracy"],
        4
    )
)


print(
    "Precision:",
    round(
        best_model["Precision"],
        4
    )
)


print(
    "Recall:",
    round(
        best_model["Recall"],
        4
    )
)


print(
    "F1 Score:",
    round(
        best_model["F1 Score"],
        4
    )
)


print(
    "ROC-AUC:",
    round(
        best_model["ROC-AUC"],
        4
    )
)


# ============================================================
# 44. SAVE MODEL COMPARISON
# ============================================================

results.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("BANK MARKETING PREDICTION PROJECT COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("1. decision_tree.png")

print("2. tuned_decision_tree.png")

print("3. model_comparison.csv")
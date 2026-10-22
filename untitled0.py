import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
    roc_auc_score,
    roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Evaluation & Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# Data & Model Training with caching
@st.cache_resource
def load_and_train_models():
    # Look for bank_marketing.csv in script directory or current directory
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_marketing.csv"),
        "bank_marketing.csv",
        os.path.join(os.getcwd(), "bank_marketing.csv")
    ]
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError("bank_marketing.csv not found in repository.")

    df = pd.read_csv(file_path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_")

    target_col = "y" if "y" in df.columns else df.columns[-1]
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = df[target_col].map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessors
    preprocessor_lr = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    preprocessor_dt = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    # Pipelines
    lr_pipe = Pipeline([
        ("prep", preprocessor_lr),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    dt_pipe = Pipeline([
        ("prep", preprocessor_dt),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    best_dt_pipe = Pipeline([
        ("prep", preprocessor_dt),
        ("clf", DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, criterion="gini", random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr_pipe.fit(X_train, y_train)
    dt_pipe.fit(X_train, y_train)
    best_dt_pipe.fit(X_train, y_train)

    # Evaluate models
    models_dict = {
        "Logistic Regression": lr_pipe,
        "Decision Tree": dt_pipe,
        "Tuned Decision Tree": best_dt_pipe
    }

    eval_results = []
    roc_data = {}

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        eval_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
            "Confusion Matrix": cm
        })
        roc_data[name] = (fpr, tpr, roc)

    eval_df = pd.DataFrame(eval_results)
    return df, X, y, models_dict, eval_df, roc_data, X_test, y_test


try:
    df, X, y, models, eval_df, roc_data, X_test, y_test = load_and_train_models()
except Exception as e:
    st.error(f"Error initializing models or loading dataset: {e}")
    st.stop()


# Navigation Tabs
st.markdown('<div class="main-header">🏦 Bank Marketing Campaign & Term Deposit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze marketing campaign efficiency, evaluate machine learning models, and predict customer deposit subscriptions in real-time.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🎯 Real-Time Customer Prediction",
    "📊 Model Evaluation & Comparison",
    "📈 Dataset & Campaign Insights"
])


# ============================================================
# TAB 1: Real-Time Prediction
# ============================================================
with tab1:
    st.subheader("Predict Customer Term Deposit Subscription")
    st.write("Fill in the customer information and marketing campaign details below:")

    col_model, col_blank = st.columns([2, 2])
    with col_model:
        selected_model_name = st.selectbox(
            "Select Prediction Model",
            options=["Tuned Decision Tree", "Decision Tree", "Logistic Regression"],
            index=0,
            help="Tuned Decision Tree is optimized with hyperparameter constraints to prevent overfitting."
        )

    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Customer Profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", options=sorted(df["job"].dropna().unique().tolist()), index=0)
            marital = st.selectbox("Marital Status", options=sorted(df["marital"].dropna().unique().tolist()))
            education = st.selectbox("Education Level", options=sorted(df["education"].dropna().unique().tolist()))
            default = st.selectbox("Has Credit in Default?", options=["no", "yes"])

        with col2:
            st.markdown("##### 💰 Financial & Account Info")
            balance = st.number_input("Average Yearly Balance (EUR)", min_value=-10000, max_value=150000, value=1500)
            housing = st.selectbox("Has Housing Loan?", options=["yes", "no"])
            loan = st.selectbox("Has Personal Loan?", options=["no", "yes"])
            contact = st.selectbox("Contact Communication Type", options=sorted(df["contact"].dropna().unique().tolist()))
            day = st.slider("Last Contact Day of Month", min_value=1, max_value=31, value=15)
            month = st.selectbox("Last Contact Month", options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=4)

        with col3:
            st.markdown("##### 📞 Campaign Details")
            duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=250, help="High duration strongly correlates with conversion.")
            campaign = st.number_input("Number of Contacts during this Campaign", min_value=1, max_value=60, value=1)
            pdays = st.number_input("Days Since Previous Contact (-1 = never)", min_value=-1, max_value=1000, value=-1)
            previous = st.number_input("Number of Contacts Before this Campaign", min_value=0, max_value=100, value=0)
            poutcome = st.selectbox("Outcome of Previous Marketing Campaign", options=sorted(df["poutcome"].dropna().unique().tolist()))

        submit_btn = st.form_submit_button("🔮 Predict Term Deposit Subscription", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        active_model = models[selected_model_name]
        pred = active_model.predict(input_data)[0]
        prob = active_model.predict_proba(input_data)[0][1] if hasattr(active_model, "predict_proba") else (1.0 if pred == 1 else 0.0)

        st.markdown("### 📋 Prediction Result")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.success("### 🎉 Result: **Subscribed (YES)**")
                st.write(f"The model predicts with **{prob * 100:.1f}% confidence** that this customer is likely to subscribe to the term deposit.")
            else:
                st.warning("### ❌ Result: **Not Subscribed (NO)**")
                st.write(f"The model predicts with **{(1 - prob) * 100:.1f}% confidence** that this customer is unlikely to subscribe.")

        with res_col2:
            st.metric(label="Subscription Probability", value=f"{prob * 100:.1f}%")
            st.progress(float(prob))


# ============================================================
# TAB 2: Model Comparison & Evaluation
# ============================================================
with tab2:
    st.subheader("Model Performance Comparison")
    
    display_df = eval_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].copy()
    st.dataframe(display_df.style.format({
        "Accuracy": "{:.3%}",
        "Precision": "{:.3%}",
        "Recall": "{:.3%}",
        "F1 Score": "{:.3%}",
        "ROC-AUC": "{:.3f}"
    }), use_container_width=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### 📊 Accuracy & ROC-AUC Comparison")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        bar_data = eval_df.melt(id_vars=["Model"], value_vars=["Accuracy", "ROC-AUC"], var_name="Metric", value_name="Score")
        sns.barplot(data=bar_data, x="Model", y="Score", hue="Metric", ax=ax_bar, palette="Blues_d")
        ax_bar.set_ylim(0.5, 1.0)
        ax_bar.set_title("Model Metrics Comparison")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_chart2:
        st.markdown("##### 📈 ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        for model_name, (fpr, tpr, roc_score) in roc_data.items():
            ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_score:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves Comparison")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown("---")
    st.markdown("##### 🧮 Confusion Matrices")
    cm_cols = st.columns(3)
    for idx, row in eval_df.iterrows():
        with cm_cols[idx]:
            st.markdown(f"**{row['Model']}**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(row["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)


# ============================================================
# TAB 3: Dataset & Campaign Insights
# ============================================================
with tab3:
    st.subheader("Bank Marketing Dataset Exploration")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Total Records", f"{df.shape[0]:,}")
    col_d2.metric("Total Features", f"{df.shape[1] - 1}")
    sub_pct = (df["y"].mean() * 100)
    col_d3.metric("Deposit Subscription Rate", f"{sub_pct:.1f}%")

    st.markdown("##### Sample Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("##### Target Variable Distribution (Subscribed vs Not)")
        fig_tgt, ax_tgt = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x="y", hue="y", palette=["#EF4444", "#10B981"], legend=False, ax=ax_tgt)
        ax_tgt.set_xticks([0, 1])
        ax_tgt.set_xticklabels(["Did Not Subscribe (0)", "Subscribed (1)"])
        ax_tgt.set_xlabel("")
        ax_tgt.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_tgt)
        plt.close(fig_tgt)

    with col_e2:
        st.markdown("##### Subscription Rate by Job Category")
        fig_job, ax_job = plt.subplots(figsize=(6, 3.5))
        job_sub = df.groupby("job")["y"].mean().sort_values(ascending=False).reset_index()
        sns.barplot(data=job_sub, x="y", y="job", hue="job", palette="viridis", legend=False, ax=ax_job)
        ax_job.set_xlabel("Subscription Rate")
        ax_job.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_job)
        plt.close(fig_job)

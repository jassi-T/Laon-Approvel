import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="LoanLens | ML Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 16px; border-radius: 14px; border: 1px solid rgba(128,128,128,.18);}
.hero {padding: 24px 26px; border-radius: 18px; background: linear-gradient(115deg,#14213d,#245b78); color: white; margin-bottom: 18px;}
.hero h1 {font-size: 2.1rem; margin: 0;}
.hero p {opacity: .85; margin: 8px 0 0 0;}
.small-note {color: #8b95a5; font-size: .88rem;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

FEATURES = ["income", "credit_score", "loan_amount", "years_working", "employment"]
TARGET = "loan_status"


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def clean_data(raw):
    df = raw.copy()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df[FEATURES + [TARGET]].copy()
    df = df.drop_duplicates()

    for col in ["income", "credit_score", "loan_amount", "years_working"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace("₹", "").str.strip(),
            errors="coerce"
        )

    df.loc[df["credit_score"] > 900, "credit_score"] = np.nan

    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()

    df = df[df[TARGET].isin(["Approved", "Rejected"])]

    employment_map = {
        "salaried": 1,
        "self-employed": 0,
        "self employed": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }

    df["employment"] = (
        df["employment"].astype(str).str.strip().str.lower().map(employment_map)
    )

    df = df.dropna(subset=[TARGET])

    return df


def build_model(data):

    X = data[FEATURES]
    y = data[TARGET]

    if y.nunique() != 2:
        raise ValueError("The loan_status column must contain both Approved and Rejected classes.")

    numeric_features = [
        "income",
        "credit_score",
        "loan_amount",
        "years_working"
    ]

    preprocess = ColumnTransformer([
        ("numeric",
         Pipeline([
             ("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())
         ]),
         numeric_features),

        ("employment",
         Pipeline([
             ("imputer", SimpleImputer(strategy="most_frequent")),
             ("scaler", StandardScaler())
         ]),
         ["employment"])
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", SVC(
            kernel="rbf",
            C=1,
            probability=True,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    return model, X_train, X_test, y_train, y_test, predictions, probabilities


st.markdown("""
<div class="hero">
  <h1>🏦 LoanLens ML Dashboard</h1>
  <p>Explore loan applications, understand the SVM model, and test a new applicant interactively.</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:

    st.header("⚙️ Dashboard controls")

    uploaded = st.file_uploader(
        "Upload loan_approval.csv",
        type=["csv"]
    )

    st.caption(
        "Required columns: income, credit_score, loan_amount, years_working, employment, loan_status"
    )

    st.divider()

    st.markdown("**Model configuration**")

    st.info("SVM · RBF kernel · C = 1 · 80/20 split · random state = 42")

    st.caption("Employment encoding: Salaried = 1 · Self-employed = 0")


if uploaded is None:

    st.info("Upload your `loan_approval.csv` file in the sidebar to launch the live dashboard.")

    st.markdown("### What you can explore")

    a, b, c = st.columns(3)

    a.metric("Model", "SVM / RBF")
    b.metric("Input features", "5")
    c.metric("Target classes", "2")

    st.markdown(
        "The dashboard will display dataset insights, model evaluation, and a live applicant prediction form."
    )

    st.stop()


try:

    raw_df = load_data(uploaded)

    df = clean_data(raw_df)

    model, X_train, X_test, y_train, y_test, y_pred, y_prob = build_model(df)

except Exception as exc:

    st.error(f"Could not prepare the dashboard: {exc}")

    st.stop()


accuracy = accuracy_score(y_test, y_pred)

class_names = ["Approved", "Rejected"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)


tab_overview, tab_explore, tab_model, tab_predict = st.tabs([
    "📊 Overview",
    "🔎 Data explorer",
    "🧠 Model performance",
    "✨ Try a prediction"
])


with tab_overview:

    st.subheader("Portfolio at a glance")

    total = len(df)

    approved = int((df[TARGET] == "Approved").sum())

    rejected = int((df[TARGET] == "Rejected").sum())

    approval_rate = approved / total * 100 if total else 0

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total applications", f"{total:,}")

    k2.metric("Approved", f"{approved:,}")

    k3.metric("Rejected", f"{rejected:,}")

    k4.metric("Approval rate", f"{approval_rate:.1f}%")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Application outcomes")

        counts = df[TARGET].value_counts().reindex(
            class_names,
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": .42}
        )

        ax.set_title("Approved vs rejected")

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    with right:

        st.markdown("#### Average applicant profile")

        summary = df.groupby(TARGET)[
            ["income", "credit_score", "loan_amount", "years_working"]
        ].mean().T

        st.dataframe(
            summary.style.format("{:,.1f}"),
            use_container_width=True
        )

    st.markdown("#### Recent rows")

    st.dataframe(
        df.tail(8),
        use_container_width=True,
        hide_index=True
    )


with tab_explore:

    st.subheader("Explore the dataset")

    f1, f2 = st.columns([1, 2])

    with f1:

        selected_feature = st.selectbox(
            "Choose a feature",
            FEATURES[:4]
        )

        show_rows = st.slider(
            "Rows to preview",
            5,
            min(50, max(5, len(df))),
            min(10, max(5, len(df)))
        )

        st.metric(
            "Missing values after cleaning",
            int(df[FEATURES].isna().sum().sum())
        )

        st.metric(
            "Duplicate rows removed",
            len(raw_df) - len(df)
        )

    with f2:

        st.markdown(
            f"#### {selected_feature.replace('_', ' ').title()} distribution"
        )

        fig, ax = plt.subplots(figsize=(7, 3.5))

        sns.histplot(
            data=df,
            x=selected_feature,
            hue=TARGET,
            kde=True,
            element="step",
            ax=ax
        )

        ax.set_xlabel(selected_feature.replace("_", " ").title())

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    st.markdown("#### Credit score vs loan amount")

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.scatterplot(
        data=df,
        x="credit_score",
        y="loan_amount",
        hue=TARGET,
        style=TARGET,
        s=75,
        ax=ax
    )

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.markdown("#### Filter applications")

    outcome_filter = st.multiselect(
        "Loan status",
        class_names,
        default=class_names
    )

    filtered = df[df[TARGET].isin(outcome_filter)]

    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} applications"
    )

    st.dataframe(
        filtered.head(show_rows),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download filtered data",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_loan_data.csv",
        "text/csv"
    )


with tab_model:

    st.subheader("How well does the SVM model perform?")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Training accuracy",
        f"{model.score(X_train, y_train) * 100:.1f}%"
    )

    m2.metric(
        "Testing accuracy",
        f"{accuracy * 100:.1f}%"
    )

    m3.metric(
        "Test records",
        f"{len(y_test)}"
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Confusion matrix")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Predicted")

        ax.set_ylabel("Actual")

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    with right:

        st.markdown("#### Classification report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    st.caption(
        "A confusion matrix shows correct and incorrect predictions for each class. Results depend on the uploaded dataset and its quality."
    )


with tab_predict:

    st.subheader("Test a new loan applicant")

    st.write(
        "Change the applicant details and ask the trained SVM model for a prediction."
    )

    with st.form("applicant_form"):

        p1, p2 = st.columns(2)

        with p1:

            income = st.number_input(
                "Annual income",
                min_value=0.0,
                value=800000.0,
                step=10000.0
            )

            credit_score = st.number_input(
                "Credit score",
                min_value=0,
                max_value=900,
                value=760,
                step=5
            )

            loan_amount = st.number_input(
                "Requested loan amount",
                min_value=0.0,
                value=300000.0,
                step=10000.0
            )

        with p2:

            years_working = st.number_input(
                "Years of work experience",
                min_value=0.0,
                max_value=60.0,
                value=6.0,
                step=1.0
            )

            employment_label = st.selectbox(
                "Employment type",
                ["Salaried", "Self-employed"]
            )

            st.caption(
                "The model converts Salaried to 1 and Self-employed to 0."
            )

        submitted = st.form_submit_button(
            "🚀 Predict loan outcome",
            use_container_width=True
        )

    if submitted:

        applicant = pd.DataFrame([{
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "years_working": years_working,
            "employment": 1 if employment_label == "Salaried" else 0
        }])

        prediction = model.predict(applicant)[0]

        probabilities = model.predict_proba(applicant)[0]

        classes = list(model.named_steps["classifier"].classes_)

        confidence = float(
            probabilities[classes.index(prediction)]
        ) * 100

        if prediction == "Approved":

            st.success(f"Predicted outcome: {prediction}")

        else:

            st.warning(f"Predicted outcome: {prediction}")

        st.metric(
            "Model confidence",
            f"{confidence:.1f}%"
        )

        prob_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        })

        fig, ax = plt.subplots(figsize=(6, 2.8))

        sns.barplot(
            data=prob_df,
            x="Outcome",
            y="Probability",
            hue="Outcome",
            legend=False,
            ax=ax
        )

        ax.set_ylim(0, 1)

        ax.set_ylabel("Probability")

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "This is an educational model demonstration, not a real lending decision or guarantee."
        )


st.divider()

st.markdown(
    '<p class="small-note">LoanLens · Educational SVM demo · Predictions should not be used as real-world credit decisions.</p>',
    unsafe_allow_html=True
)

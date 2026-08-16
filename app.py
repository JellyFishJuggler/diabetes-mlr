import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from MultiLinearRegression import MultiLinearRegression

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Diabetes MLR Studio",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------
# DESIGN TOKENS — same system as the Gradient Descent Studio,
# so the two feel like one project series rather than one-offs.
# -----------------------------------------------------
BG = "#0A0E17"
SURFACE = "#121826"
BORDER = "#232B3D"
INK = "#EDF0F5"
MUTED = "#8A93A8"
AMBER = "#F2A65A"
TEAL = "#45C7B5"

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{ font-family: {FONT_BODY}; }}
    .stApp {{ background-color: {BG}; color: {INK}; }}
    h1, h2, h3, h4 {{ font-family: {FONT_DISPLAY} !important; letter-spacing: -0.01em; }}
    section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
    div[data-testid="stMetric"] {{
        background-color: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 16px 18px;
    }}
    div[data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; color: {AMBER}; }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    .studio-chip {{
        display: inline-block; background-color: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 999px; padding: 6px 14px; margin: 4px 6px 4px 0;
        font-family: {FONT_MONO}; font-size: 0.82rem; color: {TEAL};
    }}
    .studio-card {{
        background-color: {SURFACE}; border: 1px solid {BORDER};
        border-radius: 14px; padding: 22px 24px; margin-bottom: 14px;
    }}
    .studio-eyebrow {{
        font-family: {FONT_MONO}; color: {AMBER}; font-size: 0.78rem;
        letter-spacing: 0.08em; text-transform: uppercase;
    }}
    .studio-hero-number {{
        font-family: {FONT_DISPLAY}; font-size: 3.1rem; font-weight: 700;
        color: {INK}; line-height: 1.1;
    }}
    hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


def themed_fig(fig, height=440):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT_BODY, color=INK, size=13),
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    df = df.rename(columns={
        "s1": "total_cholesterol", "s2": "ldl", "s3": "hdl",
        "s4": "tc_hdl_ratio", "s5": "triglycerides", "s6": "glucose"
    })
    return df

df = load_data()

FEATURES = ["bmi", "triglycerides", "glucose", "age"]
X = df[FEATURES]
y = df["bp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = MultiLinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test).ravel()

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)

SLR_R2 = 0.233  # BMI-only baseline from the earlier SLR project

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.markdown(f"""
<div style="padding: 6px 0 18px 0;">
    <div class="studio-eyebrow">ML from Scratch</div>
    <div style="font-family:{FONT_DISPLAY}; font-size:1.3rem; font-weight:700; color:{INK};">
        MLR Studio
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset", "Model", "SLR vs MLR", "Validation", "Conclusion"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span style='color:{MUTED}; font-size:0.85rem;'>💡 Built on the Normal Equation — "
    f"one closed-form step, no learning rate.</span>",
    unsafe_allow_html=True
)

# =====================================================
# HOME
# =====================================================
if page == "Home":
    st.markdown("<div class='studio-eyebrow'>Diabetes · Blood Pressure · Multiple Regression</div>", unsafe_allow_html=True)
    st.markdown("<div class='studio-hero-number'>Predicting Blood Pressure</div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED}; font-size:1.05rem; max-width:640px;'>"
        f"Multiple Linear Regression solved via the Normal Equation, from scratch — "
        f"benchmarked against scikit-learn, and against a single-feature baseline.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", len(X_train))
    c2.metric("Features Used", len(FEATURES))
    c3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='studio-eyebrow'>Feature set</div>", unsafe_allow_html=True)
    chips = "".join([f"<span class='studio-chip'>{c}</span>" for c in FEATURES])
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{MUTED}; font-size:0.85rem; margin-top:10px;'>"
        f"The dataset's own <code>target</code> column (disease progression score) is deliberately "
        f"excluded — it's a label for a different task, not a real-world available predictor.</p>",
        unsafe_allow_html=True
    )

# =====================================================
# DATASET
# =====================================================
elif page == "Dataset":
    st.markdown("<div class='studio-eyebrow'>Exploration</div>", unsafe_allow_html=True)
    st.markdown("## Correlation with Blood Pressure")

    tab1, tab2 = st.tabs(["Preview", "Statistical Summary"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    corr = df.drop(columns=["target"]).corr(numeric_only=True)["bp"].drop("bp").sort_values()
    colors = [AMBER if c in FEATURES else BORDER for c in corr.index]
    colors = [TEAL if (c in FEATURES and v < 0) else col for c, v, col in zip(corr.index, corr.values, colors)]

    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=colors))
    fig.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="")
    st.plotly_chart(themed_fig(fig, height=420), use_container_width=True)
    st.markdown(
        f"<p style='color:{MUTED}; font-size:0.85rem;'>Amber/teal bars are the four features used in the model; "
        f"grey bars were left out.</p>",
        unsafe_allow_html=True
    )

# =====================================================
# MODEL
# =====================================================
elif page == "Model":
    st.markdown("<div class='studio-eyebrow'>Fit & Evaluate</div>", unsafe_allow_html=True)
    st.markdown("## Model Workspace")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R² Score", f"{r2:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": model.theta.ravel()[1:]
        }).sort_values("Coefficient")
        fig = go.Figure(go.Bar(
            x=coef_df["Coefficient"], y=coef_df["Feature"], orientation="h",
            marker_color=[TEAL if v < 0 else AMBER for v in coef_df["Coefficient"]]
        ))
        fig.update_layout(title="Feature Coefficients")
        st.plotly_chart(themed_fig(fig, height=380), use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                  marker=dict(color=AMBER, size=7, opacity=0.75), name="Prediction"))
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color=MUTED, dash="dash"), name="Ideal"))
        fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual BP", yaxis_title="Predicted BP")
        st.plotly_chart(themed_fig(fig, height=380), use_container_width=True)

# =====================================================
# SLR vs MLR
# =====================================================
elif page == "SLR vs MLR":
    st.markdown("<div class='studio-eyebrow'>The Point of This Project</div>", unsafe_allow_html=True)
    st.markdown("## One Feature vs Four")

    r2 = r2_score(y_test, y_pred)
    fig = go.Figure(go.Bar(
        x=["SLR (BMI only)", "MLR (bmi, triglycerides, glucose, age)"],
        y=[SLR_R2, r2],
        marker_color=[TEAL, AMBER],
        text=[f"{SLR_R2:.3f}", f"{r2:.3f}"],
        textposition="outside",
        textfont=dict(color=INK)
    ))
    fig.update_layout(yaxis_title="R² Score", yaxis_range=[0, max(r2, SLR_R2) * 1.3])
    st.plotly_chart(themed_fig(fig, height=440), use_container_width=True)

    st.markdown(
        f"<p style='color:{MUTED};'>Adding three more real, non-leaky features on top of BMI raised R² "
        f"from <b style='color:{TEAL}'>{SLR_R2:.3f}</b> to <b style='color:{AMBER}'>{r2:.3f}</b> — "
        f"blood pressure here is explained by a combination of factors, not any single one.</p>",
        unsafe_allow_html=True
    )

# =====================================================
# VALIDATION
# =====================================================
elif page == "Validation":
    st.markdown("<div class='studio-eyebrow'>Sanity Check</div>", unsafe_allow_html=True)
    st.markdown("## Scratch vs scikit-learn")

    st.markdown(
        f"<p style='color:{MUTED};'>If the from-scratch Normal Equation is implemented correctly, "
        f"its coefficients and R² should match <code>LinearRegression</code> almost exactly.</p>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    c1.metric("Scratch R²", f"{r2_score(y_test, y_pred):.4f}")
    c2.metric("sklearn R²", f"{r2_score(y_test, sk_pred):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        "Term": ["bias/intercept"] + FEATURES,
        "Scratch": model.theta.ravel(),
        "sklearn": [sk_model.intercept_] + list(sk_model.coef_)
    })
    st.dataframe(compare_df.style.format({"Scratch": "{:.6f}", "sklearn": "{:.6f}"}), use_container_width=True)

# =====================================================
# CONCLUSION
# =====================================================
else:
    st.markdown("<div class='studio-eyebrow'>Summary</div>", unsafe_allow_html=True)
    st.markdown("## Architecture & Takeaways")

    cards = [
        ("Closed-Form Solution", "No learning rate or iterations — the Normal Equation solves for all coefficients in one matrix computation."),
        ("No Leakage", "The dataset's own progression-score column was deliberately excluded from the feature set, keeping the model realistic."),
        ("Benchmarked, Not Assumed", "Coefficients and R² match scikit-learn's LinearRegression exactly, confirming the scratch implementation is correct."),
    ]

    for title, desc in cards:
        st.markdown(f"""
        <div class="studio-card">
            <div style="font-family:{FONT_DISPLAY}; font-weight:600; font-size:1.05rem; color:{AMBER}; margin-bottom:6px;">{title}</div>
            <div style="color:{MUTED};">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
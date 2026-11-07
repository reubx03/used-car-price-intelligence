
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "used_cars_cleaned.csv"

st.set_page_config(
    page_title="AutoPulse | Used Car Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. DESIGN SYSTEM
# ==========================================

st.markdown("""
<style>

:root {
    --background: #0B1220;
    --surface: #151F32;
    --border: #26344A;
    --accent: #3B82F6;
    --text: #E8EDF5;
    --muted: #94A3B8;
}

.stApp {
    background: var(--background);
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.5px;
}

p, label {
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: #111B2D;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1 {
    font-size: 25px;
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 17px 19px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--text);
}

div[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #2563EB;
    color: white;
    border: none;
}

.result-card {
    background: #151F32;
    border: 1px solid #315D9C;
    border-radius: 14px;
    padding: 28px;
    margin-top: 24px;
}

.result-label {
    color: #93C5FD;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
}

.result-price {
    color: white;
    font-size: 38px;
    font-weight: 750;
    margin: 8px 0;
}

.section-note {
    color: var(--muted);
    font-size: 13px;
}

hr {
    border-color: var(--border);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD PROJECT FILES
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_dataset()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==========================================
# 4. PREPARE CATEGORIES
# ==========================================

df["brand"] = df["name"].str.split().str[0]

brands = sorted(df["brand"].dropna().unique().tolist())
fuel_types = sorted(df["fuel"].dropna().unique().tolist())
seller_types = sorted(df["seller_type"].dropna().unique().tolist())
transmissions = sorted(df["transmission"].dropna().unique().tolist())
owners = sorted(df["owner"].dropna().unique().tolist())

min_year = int(df["year"].min())
max_year = int(df["year"].max())
max_km = int(df["km_driven"].max())


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def format_inr(amount):

    amount = round(float(amount))
    number = str(abs(amount))

    if len(number) > 3:
        last_three = number[-3:]
        remaining = number[:-3]

        groups = []

        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]

        if remaining:
            groups.insert(0, remaining)

        number = ",".join(groups) + "," + last_three

    prefix = "-₹" if amount < 0 else "₹"

    return prefix + number


def format_lakhs(amount):
    return f"₹{amount / 100000:.2f} L"


# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🚗 AutoPulse")

    st.caption("Used Car Price Intelligence")

    st.divider()

    st.markdown("##### NAVIGATION")

    page = st.radio(
        "Choose a page",
        [
            "Price Prediction",
            "Market Insights",
            "About Model"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("Machine Learning Project")


# ==========================================
# 7. PAGE 1 — PRICE PREDICTION
# ==========================================

if page == "Price Prediction":

    st.title("Vehicle Price Estimator")

    st.markdown(
        '<p class="section-note">'
        'Enter the vehicle details to estimate its selling price.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:

            st.markdown("#### Vehicle Information")

            brand = st.selectbox(
                "Car Brand",
                brands
            )

            year = st.number_input(
                "Manufacturing Year",
                min_value=min_year,
                max_value=max_year,
                value=min(2018, max_year),
                step=1
            )

            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                max_value=max_km,
                value=min(50000, max_km),
                step=1000
            )

            fuel = st.selectbox(
                "Fuel Type",
                fuel_types
            )

        with right:

            st.markdown("#### Ownership & Transmission")

            seller_type = st.selectbox(
                "Seller Type",
                seller_types
            )

            transmission = st.selectbox(
                "Transmission",
                transmissions
            )

            owner = st.selectbox(
                "Previous Owner",
                owners
            )

        st.write("")

        submitted = st.form_submit_button(
            "Predict Selling Price",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "brand": brand
        }])

        try:

            prediction = float(model.predict(input_data)[0])

            if prediction <= 0:
                st.error("The model returned an invalid price estimate.")

            else:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            ESTIMATED SELLING PRICE
                        </div>
                        <div class="result-price">
                            {format_inr(prediction)}
                        </div>
                        <p class="section-note">
                            Estimated using the trained Random Forest model.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.markdown("#### Vehicle Summary")

                summary1, summary2, summary3 = st.columns(3)

                summary1.metric("Brand", brand)
                summary2.metric("Manufacturing Year", str(year))
                summary3.metric("Kilometres Driven", f"{km_driven:,} km")

                st.caption(
                    "This prediction is a machine learning estimate, "
                    "not a guaranteed market price. Actual selling "
                    "prices may vary."
                )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# ==========================================
# 8. PAGE 2 — MARKET INSIGHTS
# ==========================================

elif page == "Market Insights":

    st.title("Market Insights")

    st.markdown(
        '<p class="section-note">'
        'Explore patterns and trends in the used-car dataset.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    # Overall dataset KPIs

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Total Vehicles",
        f"{len(df):,}"
    )

    k2.metric(
        "Average Selling Price",
        format_lakhs(df["selling_price"].mean())
    )

    k3.metric(
        "Median Selling Price",
        format_lakhs(df["selling_price"].median())
    )

    st.caption("Summary metrics represent the complete dataset.")

    st.divider()

    # Single filter

    selected_fuel = st.selectbox(
        "Filter charts by fuel type",
        ["All"] + fuel_types
    )

    if selected_fuel == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["fuel"] == selected_fuel].copy()

    st.caption(f"{len(filtered_df):,} vehicles in the selected category")

    # Chart 1 — Price distribution

    st.markdown("### Selling Price Distribution")

    price_data = filtered_df.copy()
    price_data["Price (Lakhs)"] = (
        price_data["selling_price"] / 100000
    )

    fig1 = px.histogram(
        price_data,
        x="Price (Lakhs)",
        nbins=35,
        labels={
            "Price (Lakhs)": "Selling Price (₹ Lakhs)",
            "count": "Number of Vehicles"
        },
        color_discrete_sequence=["#3B82F6"]
    )

    fig1.update_layout(
        template="plotly_dark",
        height=370,
        showlegend=False,
        margin=dict(l=20, r=20, t=25, b=20)
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.divider()

    # Chart 2 — Average price by fuel

    st.markdown("### Average Selling Price by Fuel Type")

    fuel_summary = (
        filtered_df.groupby("fuel")
        .agg(
            average_price=("selling_price", "mean"),
            vehicle_count=("selling_price", "count")
        )
        .reset_index()
    )

    fuel_summary["Average Price (Lakhs)"] = (
        fuel_summary["average_price"] / 100000
    )

    fig2 = px.bar(
        fuel_summary,
        x="fuel",
        y="Average Price (Lakhs)",
        text="vehicle_count",
        labels={
            "fuel": "Fuel Type",
            "Average Price (Lakhs)": "Average Selling Price (₹ Lakhs)",
            "vehicle_count": "Vehicles"
        },
        color_discrete_sequence=["#3B82F6"],
        hover_data={
            "vehicle_count": True,
            "Average Price (Lakhs)": ":.2f"
        }
    )

    fig2.update_traces(
        texttemplate="%{text} vehicles",
        textposition="outside"
    )

    fig2.update_layout(
        template="plotly_dark",
        height=370,
        showlegend=False,
        margin=dict(l=20, r=20, t=25, b=20)
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.divider()

    # Chart 3 — Manufacturing year vs median price

    st.markdown("### Manufacturing Year vs Selling Price")

    year_summary = (
        filtered_df.groupby("year")
        .agg(
            median_price=("selling_price", "median"),
            vehicle_count=("selling_price", "count")
        )
        .reset_index()
    )

    year_summary["Median Price (Lakhs)"] = (
        year_summary["median_price"] / 100000
    )

    fig3 = px.line(
        year_summary,
        x="year",
        y="Median Price (Lakhs)",
        markers=True,
        labels={
            "year": "Manufacturing Year",
            "Median Price (Lakhs)": "Median Selling Price (₹ Lakhs)",
            "vehicle_count": "Vehicles"
        },
        hover_data={
            "vehicle_count": True,
            "Median Price (Lakhs)": ":.2f"
        }
    )

    fig3.update_traces(
        line=dict(color="#60A5FA", width=2.5),
        marker=dict(size=7)
    )

    fig3.update_layout(
        template="plotly_dark",
        height=370,
        showlegend=False,
        margin=dict(l=20, r=20, t=25, b=20)
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.caption(
        "All visualizations are calculated from the cleaned dataset. "
        "Small category counts may affect the reliability of averages "
        "and trends."
    )


# ==========================================
# 9. PAGE 3 — ABOUT MODEL
# ==========================================

elif page == "About Model":

    st.title("About the Model")

    st.markdown(
        '<p class="section-note">'
        'Understanding the machine learning approach behind AutoPulse.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("### Project Objective")

    st.write(
        "The objective is to estimate the selling price of used cars "
        "based on vehicle characteristics such as manufacturing year, "
        "kilometres driven, fuel type, transmission and ownership."
    )

    st.divider()

    st.markdown("### Machine Learning Approach")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem Type**")
        st.write("Supervised Learning — Regression")

    with col2:
        st.markdown("**Dataset Size**")
        st.write(f"{len(df):,} cleaned vehicle records")

    st.write("")

    st.markdown("### Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ],
        "MAE (₹)": [
            179642.89,
            198734.06,
            158827.69
        ],
        "RMSE (₹)": [
            384722.69,
            473873.48,
            367979.71
        ],
        "R² Score": [
            0.54,
            0.30,
            0.58
        ]
    })

    comparison["MAE (₹)"] = comparison["MAE (₹)"].map(
        lambda x: f"₹{x:,.2f}"
    )

    comparison["RMSE (₹)"] = comparison["RMSE (₹)"].map(
        lambda x: f"₹{x:,.2f}"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    st.markdown("### Selected Model")

    st.success("Random Forest Regressor")

    st.write(
        "Random Forest was selected because it achieved the lowest "
        "MAE and RMSE and the highest R² score among the three "
        "evaluated models."
    )

    st.caption(
        "Test-set performance: MAE ₹1.59 lakh | "
        "RMSE ₹3.68 lakh | R² 0.58. "
        "These results indicate moderate predictive performance."
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.write(
        "Python · Pandas · NumPy · Scikit-learn · "
        "Streamlit · Plotly · Joblib"
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<p class="section-note" style="text-align:center;">'
    'AutoPulse · Used Car Price Intelligence · '
    'Machine Learning Regression Project'
    '</p>',
    unsafe_allow_html=True
)

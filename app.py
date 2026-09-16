import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Personal Carbon Footprint",
    page_icon="🌱",
    layout="wide")
    

# =========================================================
# DASHBOARD HEADER
# =========================================================

st.title("🌱 Personal Carbon Footprint Dashboard")
st.markdown(
    "This dashboard predicts your daily carbon footprint "
    "using a machine learning model based on your lifestyle activities."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

possible_files = [
    "personal_carbon_footprint_behavior.csv",
    "personal_carbon_footprint_behaviour.csv"
]

data_file = None

for file in possible_files:
    if Path(file).exists():
        data_file = file
        break

if data_file is None:
    st.error("Dataset CSV file not found. Please keep the CSV in the same folder as app.py.")
    st.stop()

df = pd.read_csv(data_file)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model_file = "carbon_footprint_model.pkl"

if not Path(model_file).exists():
    st.error("carbon_footprint_model.pkl not found. Please keep the model file in the same folder as app.py.")
    st.stop()

model = joblib.load(model_file)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌱 Carbon Footprint")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Insights",
        "📈 Visualizations",
        "🔮 Carbon Prediction",
        "🌱 Recommendations",
        "🤖 Model Performance",
        "📅 Future Projection"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🌱 Personal Carbon Footprint Prediction")
    st.subheader("Machine Learning Based Sustainability Dashboard")

    st.write(
        """
        This project analyzes personal lifestyle and behavioural data
        to estimate an individual's carbon footprint using Machine Learning.
        
        The system analyzes transportation, electricity consumption,
        renewable energy usage, food habits, screen time, waste generation
        and eco-friendly actions.
        """
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📋 Total Records", len(df))

    col2.metric(
        "🌍 Average Carbon",
        f"{df['carbon_footprint_kg'].mean():.2f} kg"
    )

    col3.metric(
        "⬇️ Minimum",
        f"{df['carbon_footprint_kg'].min():.2f} kg"
    )

    col4.metric(
        "⬆️ Maximum",
        f"{df['carbon_footprint_kg'].max():.2f} kg"
    )

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Predict personal carbon footprint using Machine Learning.",
        "Analyze emissions based on transport, energy, food and waste.",
        "Compare different Machine Learning models.",
        "Identify factors affecting carbon footprint.",
        "Provide personalized sustainability recommendations.",
        "Develop an interactive Streamlit dashboard."
    ]

    for item in objectives:
        st.write("•", item)

    st.success(
        "The project uses XGBoost as the final selected model because it achieved the highest R² score."
    )


# =========================================================
# DATASET INSIGHTS
# =========================================================

elif page == "📊 Dataset Insights":

    st.title("📊 Dataset Insights")

    st.write(
        "The dataset contains personal behavioural information used to analyze and predict carbon footprint."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🌍 Carbon Impact Distribution")

    impact_counts = df["carbon_impact_level"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        impact_counts.index,
        impact_counts.values
    )

    ax.set_xlabel("Carbon Impact Level")
    ax.set_ylabel("Number of Records")
    ax.set_title("Carbon Impact Level Distribution")

    st.pyplot(fig)


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "📈 Visualizations":

    st.title("📈 Carbon Footprint Analysis")

    # -----------------------------------------------------
    # Transport
    # -----------------------------------------------------

    st.subheader("🚗 Carbon Footprint by Transport Mode")

    transport_avg = (
        df.groupby("transport_mode")["carbon_footprint_kg"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots()

    ax.bar(
        transport_avg.index,
        transport_avg.values
    )

    ax.set_xlabel("Transport Mode")
    ax.set_ylabel("Average Carbon Footprint (kg)")
    ax.set_title("Carbon Footprint by Transport Mode")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------------------------------
    # Food
    # -----------------------------------------------------

    st.subheader("🍽️ Carbon Footprint by Food Type")

    food_avg = (
        df.groupby("food_type")["carbon_footprint_kg"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots()

    ax.bar(
        food_avg.index,
        food_avg.values
    )

    ax.set_xlabel("Food Type")
    ax.set_ylabel("Average Carbon Footprint (kg)")
    ax.set_title("Carbon Footprint by Food Type")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------------------------------
    # Day Type
    # -----------------------------------------------------

    st.subheader("📅 Weekday vs Weekend")

    day_avg = (
        df.groupby("day_type")["carbon_footprint_kg"]
        .mean()
    )

    fig, ax = plt.subplots()

    ax.bar(
        day_avg.index,
        day_avg.values
    )

    ax.set_xlabel("Day Type")
    ax.set_ylabel("Average Carbon Footprint (kg)")
    ax.set_title("Weekday vs Weekend Carbon Footprint")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------------------------------
    # Impact Level
    # -----------------------------------------------------

    st.subheader("🌍 Carbon Footprint by Impact Level")

    impact_avg = (
        df.groupby("carbon_impact_level")["carbon_footprint_kg"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots()

    ax.bar(
        impact_avg.index,
        impact_avg.values
    )

    ax.set_xlabel("Impact Level")
    ax.set_ylabel("Average Carbon Footprint (kg)")
    ax.set_title("Average Carbon Footprint by Impact Level")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------------------------------
    # Correlation Heatmap
    # -----------------------------------------------------

    st.subheader("🔥 Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(
        correlation,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90
    )

    ax.set_yticklabels(correlation.columns)

    ax.set_title("Correlation Heatmap")

    fig.colorbar(image)

    st.pyplot(fig)


# =========================================================
# CARBON PREDICTION
# =========================================================

elif page == "🔮 Carbon Prediction":

    st.title("🔮 Carbon Footprint Prediction")

    st.write(
        "Enter your daily lifestyle information to estimate your carbon footprint."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        day_type = st.selectbox(
            "📅 Day Type",
            sorted(df["day_type"].unique())
        )

        transport_mode = st.selectbox(
            "🚗 Transport Mode",
            sorted(df["transport_mode"].unique())
        )

        distance_km = st.number_input(
            "🛣️ Daily Travel Distance (km)",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

        electricity_kwh = st.number_input(
            "⚡ Daily Electricity Usage (kWh)",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

        renewable_usage_pct = st.number_input(
            "☀️ Renewable Energy Usage (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=5.0
        )

    with col2:

        food_type = st.selectbox(
            "🍽️ Food Type",
            sorted(df["food_type"].unique())
        )

        screen_time_hours = st.number_input(
            "💻 Daily Screen Time (hours)",
            min_value=0.0,
            value=5.0,
            step=0.5
        )

        waste_generated_kg = st.number_input(
            "🗑️ Daily Waste Generated (kg)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        eco_actions = st.number_input(
            "🌱 Number of Eco Actions",
            min_value=0,
            value=2,
            step=1
        )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Carbon Footprint",
        width="stretch"
    )

    if predict_button:

        input_data = pd.DataFrame({
            "day_type": [day_type],
            "transport_mode": [transport_mode],
            "distance_km": [distance_km],
            "electricity_kwh": [electricity_kwh],
            "renewable_usage_pct": [renewable_usage_pct],
            "food_type": [food_type],
            "screen_time_hours": [screen_time_hours],
            "waste_generated_kg": [waste_generated_kg],
            "eco_actions": [eco_actions]
        })

        try:

            prediction = model.predict(input_data)[0]

            prediction = max(0, float(prediction))

            st.markdown("---")

            st.subheader("🌍 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Predicted Carbon Footprint",
                    f"{prediction:.2f} kg"
                )

            with result_col2:

                if prediction < 6:
                    impact = "LOW"
                    st.success("🟢 Carbon Impact: LOW")

                elif prediction < 10:
                    impact = "MEDIUM"
                    st.warning("🟡 Carbon Impact: MEDIUM")

                else:
                    impact = "HIGH"
                    st.error("🔴 Carbon Impact: HIGH")

            st.info(
                f"Your estimated daily carbon footprint is approximately {prediction:.2f} kg."
            )

        except Exception as e:

            st.error("Prediction could not be generated.")

            st.code(str(e))


# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "🌱 Recommendations":

    st.title("🌱 Sustainability Recommendations")

    st.write(
        """
        Small lifestyle changes can significantly reduce personal carbon emissions.
        Below are recommendations based on the major factors analyzed in this project.
        """
    )

    st.markdown("---")

    st.subheader("🚗 Transportation")

    st.write(
        "• Prefer walking, cycling or public transport for short-distance travel."
    )

    st.write(
        "• Reduce unnecessary car trips and consider electric vehicles where possible."
    )

    st.markdown("---")

    st.subheader("⚡ Energy")

    st.write(
        "• Reduce unnecessary electricity consumption."
    )

    st.write(
        "• Switch off lights and electronic devices when they are not required."
    )

    st.write(
        "• Increase the use of renewable energy."
    )

    st.markdown("---")

    st.subheader("🍽️ Food")

    st.write(
        "• Increasing plant-based food choices can help reduce carbon emissions."
    )

    st.write(
        "• Avoid unnecessary food waste."
    )

    st.markdown("---")

    st.subheader("🗑️ Waste")

    st.write(
        "• Reduce single-use products."
    )

    st.write(
        "• Reuse and recycle materials whenever possible."
    )

    st.markdown("---")

    st.subheader("🌱 Eco-Friendly Actions")

    st.write(
        "• Plant trees and support environmental initiatives."
    )

    st.write(
        "• Use reusable bags, bottles and containers."
    )

    st.success(
        "🌍 Sustainable habits today can create a cleaner future tomorrow!"
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        "Three Machine Learning models were evaluated using MAE, MSE and R²."
    )

    st.markdown("---")

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ],

        "MAE": [
            0.4714,
            0.5037,
            0.2598
        ],

        "MSE": [
            0.3909,
            0.4354,
            0.1255
        ],

        "R² Score": [
            0.9482,
            0.9423,
            0.9834
        ]
    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🏆 Best Performing Model")

    st.success(
        "XGBoost achieved the highest R² score of approximately 0.9834 and the lowest MAE and MSE."
    )

    st.write(
        """
        R² indicates how well the model explains the variation in carbon footprint.
        A value close to 1 indicates strong predictive performance.
        """
    )

    st.markdown("---")

    st.subheader("📊 R² Score Comparison")

    fig, ax = plt.subplots()

    ax.bar(
        results["Model"],
        results["R² Score"]
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("R² Score")
    ax.set_title("Machine Learning Model Comparison")

    st.pyplot(fig)


# =========================================================
# FUTURE PROJECTION
# =========================================================

elif page == "📅 Future Projection":

    st.title("📅 Future Carbon Emission Projection")

    st.write(
        """
        This section estimates future emissions based on the predicted daily
        carbon footprint. It provides a simple projection for longer periods.
        """
    )

    st.markdown("---")

    daily_value = st.number_input(
        "Enter estimated daily carbon footprint (kg)",
        min_value=0.0,
        value=float(df["carbon_footprint_kg"].mean()),
        step=0.1
    )

    days = st.slider(
        "Projection Period (days)",
        min_value=7,
        max_value=365,
        value=30
    )

    total_emission = daily_value * days

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Daily",
        f"{daily_value:.2f} kg"
    )

    col2.metric(
        "Monthly Equivalent",
        f"{daily_value * 30:.2f} kg"
    )

    col3.metric(
        f"{days}-Day Projection",
        f"{total_emission:.2f} kg"
    )

    st.markdown("---")

    projection_days = np.arange(1, days + 1)

    projection_values = daily_value * projection_days

    fig, ax = plt.subplots()

    ax.plot(
        projection_days,
        projection_values
    )

    ax.set_xlabel("Number of Days")
    ax.set_ylabel("Cumulative Carbon Emission (kg)")
    ax.set_title("Projected Cumulative Carbon Emissions")

    st.pyplot(fig)

    st.info(
        "This is a scenario-based future projection because the dataset does not contain actual calendar-date/time-series observations."
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "🌱 Personal Carbon Footprint Prediction Project"
)

st.sidebar.caption(
    "Developed using Python, Pandas, Scikit-learn, XGBoost and Streamlit."
)
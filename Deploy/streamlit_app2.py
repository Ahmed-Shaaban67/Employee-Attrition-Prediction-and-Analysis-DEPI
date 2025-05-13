import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Load model and preprocessor
try:
    model = joblib.load("xgboost_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()

# Manual Entry Form
st.subheader("Manual Entry")
with st.form(key="employee_form"):
    col1, col2 = st.columns(2)
    with col1:
        employee_number = st.text_input("Employee Number", value="EMP001")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4, value=3)
        job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=4, value=4)
        stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value=3, value=0)
        department = st.selectbox("Department", options=["Sales", "R&D", "HR"], index=0)
    with col2:
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=years_at_company, value=2)
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=years_at_company, value=2)
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0, max_value=100, value=5)
        monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=5000)
        overtime = st.selectbox("OverTime", options=["No", "Yes"], index=0)
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"], index=0)
    submit_button = st.form_submit_button(label="Submit")

# File Upload
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Tabs
tabs = st.tabs(["Data Overview", "Predictions", "At-Risk Employees", "Dashboard"])

# Prediction Logic
if submit_button or uploaded_file:
    if submit_button:
        # Prepare data from manual entry
        data = pd.DataFrame({
            "EmployeeNumber": [employee_number],
            "Age": [age],
            "JobInvolvement": [job_involvement],
            "JobLevel": [job_level],
            "JobSatisfaction": [job_satisfaction],
            "StockOptionLevel": [stock_option_level],
            "YearsAtCompany": [years_at_company],
            "YearsInCurrentRole": [years_in_current_role],
            "YearsWithCurrManager": [years_with_curr_manager],
            "DistanceFromHome": [distance_from_home],
            "MonthlyIncome": [monthly_income],
            "OverTime": [overtime],
            "Department": [department],
            "MaritalStatus": [marital_status]
        })
    elif uploaded_file:
        # Load data from uploaded CSV
        data = pd.read_csv(uploaded_file)
        # Ensure required columns exist
        required_columns = ["EmployeeNumber", "Age", "JobInvolvement", "JobLevel", "JobSatisfaction", 
                           "StockOptionLevel", "YearsAtCompany", "YearsInCurrentRole", 
                           "YearsWithCurrManager", "DistanceFromHome", "MonthlyIncome", "OverTime"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing columns in CSV: {missing_columns}")
            st.stop()

    # Create TenureCategory and SalaryBand
    data["TenureCategory"] = pd.cut(data["YearsAtCompany"], bins=[-float("inf"), 2, 5, float("inf")], labels=["Short", "Medium", "Long"])
    data["SalaryBand"] = pd.cut(data["MonthlyIncome"], bins=[-float("inf"), 3000, 6000, float("inf")], labels=["Low", "Medium", "High"])

    # Define features expected by the preprocessor
    model_features = ["Age", "JobInvolvement", "JobLevel", "JobSatisfaction", 
                      "StockOptionLevel", "YearsAtCompany", "YearsInCurrentRole", 
                      "YearsWithCurrManager", "DistanceFromHome", "MonthlyIncome", 
                      "OverTime", "TenureCategory", "SalaryBand"]

    # Transform the data
    try:
        # Select only the features needed for the model
        X = data[model_features]
        transformed_data = preprocessor.transform(X)
        probabilities = model.predict_proba(transformed_data)[:, 1]
        predictions = ["Yes" if prob >= 0.5 else "No" for prob in probabilities]

        # Add predictions to data
        data["Attrition_Probability"] = probabilities
        data["Attrition_Prediction"] = predictions

        # Data Overview
        with tabs[0]:
            st.write("### Data Overview")
            st.write("Processed data shape:", transformed_data.shape)
            st.dataframe(data)

        # Predictions
        with tabs[1]:
            st.write("### Predictions")
            st.dataframe(data[["EmployeeNumber", "Attrition_Probability", "Attrition_Prediction"]])

        # At-Risk Employees
        with tabs[2]:
            st.write("### At-Risk Employees")
            at_risk = data[data["Attrition_Probability"] > 0.6]
            if not at_risk.empty:
                st.dataframe(at_risk[["EmployeeNumber", "Attrition_Probability", "Attrition_Prediction"]])
            else:
                st.write("No at-risk employees found.")

        # Dashboard
        with tabs[3]:
            st.subheader("Interactive Visualizations & Insights")

            # Display total and at-risk employee counts
            total_employees = len(data)
            at_risk_employees = len(at_risk)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Employees", value=total_employees)
            with col2:
                st.metric(label="At-Risk Employees", value=at_risk_employees, 
                         delta=f"{(at_risk_employees/total_employees)*100:.1f}% of total")

            # Check if Department and MaritalStatus are available
            has_department = "Department" in data.columns
            has_marital_status = "MaritalStatus" in data.columns

            if not (has_department and has_marital_status):
                st.warning("Dashboard visualizations require 'Department' and 'MaritalStatus' columns. "
                          "Please ensure these columns are included in the CSV or use manual entry.")
            else:
                # Interactive filters for visualizations
                col1, col2, col3 = st.columns(3)
                with col1:
                    dept_options = ["All"] + sorted(data['Department'].dropna().unique())
                    selected_dept = st.selectbox("Department (Dashboard)", dept_options, key="dash_dept")
                with col2:
                    ms_options = ["All"] + sorted(data['MaritalStatus'].dropna().unique())
                    selected_ms = st.selectbox("Marital Status (Dashboard)", ms_options, key="dash_ms")
                with col3:
                    ot_options = ["All"] + sorted(data['OverTime'].dropna().unique())
                    selected_ot = st.selectbox("OverTime (Dashboard)", ot_options, key="dash_ot")

                dashboard_filtered = data.copy()
                if selected_dept != "All":
                    dashboard_filtered = dashboard_filtered[dashboard_filtered['Department'] == selected_dept]
                if selected_ms != "All":
                    dashboard_filtered = dashboard_filtered[dashboard_filtered['MaritalStatus'] == selected_ms]
                if selected_ot != "All":
                    dashboard_filtered = dashboard_filtered[dashboard_filtered['OverTime'] == selected_ot]

                # Handle empty or whitespace categories
                dashboard_filtered['Attrition_Prediction'] = dashboard_filtered['Attrition_Prediction'].fillna('No').replace('', 'No')

                # Define consistent category order and colors
                attrition_order = ['No', 'Yes']
                color_map = {'No': '#2ecc71', 'Yes': '#e74c3c'}  # Green for No, Red for Yes

                # 1. Attrition Percentage by Department (Stacked Bar)
                dept_counts = dashboard_filtered.groupby(["Department", "Attrition_Prediction"]).size().reset_index(name='Count')
                dept_total = dashboard_filtered.groupby("Department").size().reset_index(name='Total')
                dept_attrition = pd.merge(dept_counts, dept_total, on="Department")
                dept_attrition["Percent"] = dept_attrition["Count"] / dept_attrition["Total"] * 100

                if not dept_attrition.empty:
                    fig1 = px.bar(
                        dept_attrition,
                        x="Department",
                        y="Percent",
                        color="Attrition_Prediction",
                        barmode="stack",
                        text="Percent",
                        title="Attrition Percentage by Department (Interactive)",
                        color_discrete_map=color_map,
                        category_orders={"Attrition_Prediction": attrition_order}
                    )
                    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No data to display for Department chart.")

                # 2. Attrition by Marital Status (Count)
                if dashboard_filtered["MaritalStatus"].nunique() > 0:
                    fig2 = px.histogram(
                        dashboard_filtered,
                        x="MaritalStatus",
                        color="Attrition_Prediction",
                        barmode="group",
                        title="Attrition by Marital Status",
                        color_discrete_map=color_map,
                        category_orders={"Attrition_Prediction": attrition_order}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data to display for Marital Status chart.")

                # 3. Attrition by OverTime (Stacked Percentage)
                if dashboard_filtered["OverTime"].nunique() > 0 and dashboard_filtered["Attrition_Prediction"].nunique() > 0:
                    cross_tab = pd.crosstab(dashboard_filtered["OverTime"], dashboard_filtered["Attrition_Prediction"], normalize='index') * 100
                    cross_tab = cross_tab.reset_index().melt(id_vars='OverTime', var_name='Attrition_Prediction', value_name='Percent')
                    fig3 = px.bar(
                        cross_tab,
                        x='OverTime',
                        y='Percent',
                        color='Attrition_Prediction',
                        barmode='stack',
                        title="Employee Attrition by Overtime (Stacked Percentage)",
                        color_discrete_map=color_map,
                        category_orders={"Attrition_Prediction": attrition_order}
                    )
                    fig3.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No data to display for OverTime chart.")

                # 4. Distribution of Attrition Probability
                if dashboard_filtered["Attrition_Prediction"].nunique() > 0:
                    fig4 = px.histogram(
                        dashboard_filtered,
                        x="Attrition_Probability",
                        nbins=20,
                        color="Attrition_Prediction",
                        title="Attrition Probability Distribution",
                        color_discrete_map=color_map,
                        category_orders={"Attrition_Prediction": attrition_order}
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No data to display for Attrition Probability chart.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please submit manual entry or upload a CSV file to see results.")
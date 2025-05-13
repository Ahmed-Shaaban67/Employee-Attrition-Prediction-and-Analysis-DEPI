import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Employee Attrition Prediction Dashboard", layout="wide")
st.title("Employee Attrition Prediction Dashboard")
st.write("Upload a test CSV file to predict employee attrition, analyze risks, and get recommendations.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def load_model_and_preprocessor():
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = joblib.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = joblib.load(f)
        return model, preprocessor
    except:
        return None, None

def get_possible_reason(employee_data, monthly_income_mean):
    reasons = []
    if employee_data['OverTime'].iloc[0] == 'Yes':
        reasons.append("Working overtime")
    if employee_data['MonthlyIncome'].iloc[0] < monthly_income_mean * 0.8:
        reasons.append("Low monthly income")
    if employee_data['YearsAtCompany'].iloc[0] < 3:
        reasons.append("Short tenure at company")
    if employee_data['DistanceFromHome'].iloc[0] > 10:
        reasons.append("Long commute distance")
    return reasons if reasons else ["No specific reason identified"]

def get_recommendations(reasons):
    recommendations = []
    for reason in reasons:
        if reason == "Working overtime":
            recommendations.append("Reduce overtime hours or offer flexible schedules")
        elif reason == "Low monthly income":
            recommendations.append("Consider salary adjustments or bonuses")
        elif reason == "Short tenure at company":
            recommendations.append("Implement better onboarding and retention programs")
        elif reason == "Long commute distance":
            recommendations.append("Offer remote work options or transportation benefits")
    return recommendations if recommendations else ["No specific recommendations"]

def preprocess_data(data):
    """Apply the same feature engineering as in the notebook."""
    # Create TenureCategory
    data['TenureCategory'] = pd.cut(
        data['YearsAtCompany'],
        bins=[-float('inf'), 2, 5, float('inf')],
        labels=['Short', 'Medium', 'Long']
    )

    # Create SalaryBand
    data['SalaryBand'] = pd.cut(
        data['MonthlyIncome'],
        bins=[-float('inf'), 3000, 6000, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return data

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Apply feature engineering to create missing columns
    data = preprocess_data(data)

    # Handle missing values for categorical columns
    categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
        'MaritalStatus', 'Over18', 'OverTime', 'TenureCategory', 'SalaryBand'
    ]
    for col in categorical_cols:
        if col in data.columns:
            mode = data[col].mode()
            if not mode.empty:
                data[col] = data[col].fillna(mode[0])
            else:
                data[col] = data[col].fillna('Unknown')

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", "Predictions", "At-Risk Employees", "Filters", "Dashboard"
    ])

    with tab1:
        st.subheader("Data Preview")
        st.dataframe(data.head())

    model, preprocessor = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.error("No trained model or preprocessor found. Please ensure xgboost_model.pkl and preprocessor.pkl are in the project directory.")
    else:
        # Define the features expected by the model
        final_selected_features = [
            'Age', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'StockOptionLevel',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager', 'DistanceFromHome',
            'MonthlyIncome', 'TenureCategory', 'SalaryBand'
        ]

        # Check for missing columns
        missing_cols = [col for col in final_selected_features if col not in data.columns]
        if missing_cols:
            st.error(f"The following required columns are missing in the uploaded data: {missing_cols}")
        else:
            X = data[final_selected_features]
            X_processed = preprocessor.transform(X)
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1]
            data['Attrition_Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
            data['Attrition_Probability'] = probabilities

            monthly_income_mean = data['MonthlyIncome'].mean() if 'MonthlyIncome' in data.columns else 0
            at_risk = data[data['Attrition_Probability'] > 0.7].copy()

            # ==== Tab 2: Predictions with Yes filter and recommendations ====
            with tab2:
                st.subheader("All Predictions")
                yes_data = data[data['Attrition_Prediction'] == 'Yes']
                st.write(f"Number of employees predicted to leave: {len(yes_data)}")
                st.dataframe(yes_data[['EmployeeNumber', 'Attrition_Probability']])

                if not yes_data.empty:
                    selected_emp = st.selectbox(
                        "Select Employee Number (Predicted Yes)", 
                        yes_data['EmployeeNumber'],
                        key="yes_emp"
                    )
                    emp_row = yes_data[yes_data['EmployeeNumber'] == selected_emp].iloc[0]
                    st.write(f"*Attrition Probability:* {emp_row['Attrition_Probability']*100:.1f}%")

                    # Calculate reasons and recommendations
                    reasons = get_possible_reason(pd.DataFrame([emp_row]), monthly_income_mean)
                    recommendations = get_recommendations(reasons)
                    st.write("*Possible Reasons:*")
                    for r in reasons:
                        st.write(f"- {r}")
                    st.write("*Recommendations:*")
                    for rec in recommendations:
                        st.write(f"- {rec}")
                else:
                    st.info("No employees predicted as 'Yes' for attrition.")

                st.subheader("Download Results")
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv"
                )

            # ==== Tab 3: At-Risk Employees ====
            with tab3:
                st.subheader("Employees at High Risk of Attrition")
                if not at_risk.empty:
                    at_risk['Possible_Reasons'] = at_risk.apply(
                        lambda row: ", ".join(get_possible_reason(pd.DataFrame([row]), monthly_income_mean)), axis=1)
                    at_risk['Recommendations'] = at_risk['Possible_Reasons'].apply(
                        lambda x: ", ".join(get_recommendations(x.split(", "))))
                    st.dataframe(at_risk[['EmployeeNumber', 'Attrition_Probability', 'Possible_Reasons', 'Recommendations']])
                    high_risk_count = len(at_risk[at_risk['Attrition_Probability'] > 0.9])
                    if high_risk_count > 0:
                        st.error(f"Alert: {high_risk_count} employees have a very high attrition risk (>90%)!")
                    elif len(at_risk) > 5:
                        st.warning(f"Warning: {len(at_risk)} employees are at high risk of attrition.")
                else:
                    st.write("No employees identified as high risk.")

            # ==== Tab 4: Filters ====
            with tab4:
                st.subheader("Filter At-Risk Employees")
                col1, col2, col3 = st.columns(3)
                with col1:
                    dept_options = ['All'] + sorted(data['Department'].dropna().unique())
                    department = st.selectbox("Select Department", options=dept_options)
                with col2:
                    min_age = int(data['Age'].min()) if not data['Age'].isna().all() else 0
                    max_age = int(data['Age'].max()) if not data['Age'].isna().all() else 0
                    if min_age < max_age:
                        age_range = st.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
                    else:
                        age_range = (min_age, max_age)
                        st.info(f"All employees have the same age: {min_age}")
                with col3:
                    jr_options = ['All'] + sorted(data['JobRole'].dropna().unique())
                    job_role = st.selectbox("Select Job Role", options=jr_options)

                filtered_data = at_risk.copy()
                if department != 'All':
                    filtered_data = filtered_data[filtered_data['Department'] == department]
                filtered_data = filtered_data[(filtered_data['Age'] >= age_range[0]) & (filtered_data['Age'] <= age_range[1])]
                if job_role != 'All':
                    filtered_data = filtered_data[filtered_data['JobRole'] == job_role]

                if not filtered_data.empty:
                    st.write("Filtered At-Risk Employees:")
                    st.dataframe(filtered_data[['EmployeeNumber', 'Attrition_Probability', 'Possible_Reasons', 'Recommendations', 'Department', 'Age', 'JobRole']])
                else:
                    st.write("No employees match the selected filters.")

            # ==== Tab 5: Dashboard with clear colors ====
            with tab5:
                st.subheader("Interactive Visualizations & Insights")

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
                    st.info("No data to display for this chart.")

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
    st.info("Please upload your test data CSV file to start.")
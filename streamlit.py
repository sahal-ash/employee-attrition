import streamlit as st
import pickle
from PIL import Image
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

def main():
    # Load the model and scaler
    try:
        scaler = pickle.load(open('scaler.sav', 'rb'))
        model = pickle.load(open('model.sav', 'rb'))
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'scaler.sav' and 'model.sav' are in the correct directory.")
        return

    # Page title and header image
    st.title(":bar_chart: **Employee Attrition Prediction System**")
    try:
        img = Image.open('pic1.jpg')
        st.image(img, caption="Analyze Employee Attrition with ML", width=850)
    except FileNotFoundError:
        st.warning("Image file 'pic1.jpg' not found. Please ensure the file exists in the correct path.")

    # Description
    st.markdown(
        """
        Welcome to the **Employee Attrition Prediction System**. 
        This tool is designed to help HR teams and management predict whether employees are likely to leave the organization, 
        based on various personal and professional factors. The prediction uses a trained machine learning model.

        **How it works:**
        - Input relevant details about an employee.
        - Click on the `Predict` button to view the result.
        - Use the insights to make informed decisions regarding employee retention.

        **Note:** All data is hypothetical and for demonstration purposes only.
        """
    )
    # Footer section with Colab link
    st.markdown(
        "### 🔗 Resources\n"
        "You can access the Colab notebook for this project using the following link:\n\n"
        "[Open Colab Notebook](https://colab.research.google.com/drive/1EydQ7qt0k-dxMPYGv8hzzQyIW57wuU2C)"
    )

    # Input sections
    with st.expander("Employee Details"):
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"],
                                  help="The department the employee belongs to.")
        DistanceFromHome = st.number_input("Distance From Home", min_value=1, max_value=50,
                                           help="Enter the distance from the employee's home to the workplace in kilometers.")

    with st.expander("Personal Details"):
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"],
                                     help="Select the marital status of the employee.")
        Gender = st.selectbox("Gender", ["Male", "Female"], help="The employee's gender.")
        Age = st.number_input("Age", min_value=18, max_value=100, help="Enter the employee's age.")

    with st.expander("Job and Compensation Details"):
        BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                      help="How frequently the employee travels for business.")
        JobRole = st.selectbox("Job Role",
                               ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Other"],
                               help="Employee's current job role.")
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5], help="The job seniority level.")
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000,
                                        help="The employee's monthly income in dollars.")
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3],
                                        help="Stock option level offered to the employee.")
        OverTime = st.selectbox("Over Time", ["Yes", "No"], help="Whether the employee works overtime.")
        PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, max_value=100,
                                            help="The percentage salary hike.")

    with st.expander("Performance and Satisfaction"):
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], help="Rate the employee's job satisfaction.")
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                               help="Rate the employee's work environment satisfaction.")
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4],
                                                help="Rate the employee's satisfaction with their relationships.")
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4], help="Rate the employee's job involvement.")
        PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0,
                                                help="The number of training sessions attended last year.")
        WorkLifeBalance = st.selectbox("Work Life Balance", [1, 2, 3, 4],
                                       help="Rate the employee's work-life balance (1 = poor, 4 = excellent).")

    with st.expander("Career and Education"):
        Education = st.selectbox("Education Level", [1, 2, 3, 4, 5], help="Highest education level of the employee.")
        EducationField = st.selectbox("Education Field",
                                      ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources",
                                       "Other"], help="Field of study of the employee.")
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, help="Total years of work experience.")
        YearsAtCompany = st.number_input("Years At Company", min_value=0,
                                         help="Total years the employee has been in the company.")
        YearsInCurrentRole = st.number_input("Years In Current Role", min_value=0,
                                             help="Years the employee has been in their current role.")
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0,
                                                  help="Years since the employee's last promotion.")
        YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0,
                                               help="Years the employee has worked with their current manager.")
        DailyRate = st.number_input("Daily Rate", min_value=0, help="Daily rate of the employee.")
        HourlyRate = st.number_input("Hourly Rate", min_value=0, help="Hourly rate of the employee.")
        MonthlyRate = st.number_input("Monthly Rate", min_value=0, help="Monthly rate of the employee.")
        NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0,
                                             help="Number of companies worked by the employee.")

    # Encode categorical variables
    feature_dict = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
        'EducationField': {'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3,
                           'Human Resources': 4, 'Other': 5},
        'Gender': {'Male': 0, 'Female': 1},
        'JobRole': {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manager': 3,
                    'Other': 4},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'OverTime': {'Yes': 1, 'No': 0},
        'WorkLifeBalance': {1: 1, 2: 2, 3: 3, 4: 4}
    }

    # Feature vector
    features = [
        Age, feature_dict['BusinessTravel'][BusinessTravel], DailyRate,
        feature_dict['Department'][Department], DistanceFromHome, Education,
        feature_dict['EducationField'][EducationField], EnvironmentSatisfaction,
        feature_dict['Gender'][Gender], HourlyRate, JobInvolvement, JobLevel,
        feature_dict['JobRole'][JobRole], JobSatisfaction,
        feature_dict['MaritalStatus'][MaritalStatus], MonthlyIncome, MonthlyRate,
        NumCompaniesWorked, feature_dict['OverTime'][OverTime], PercentSalaryHike,
        PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
        TrainingTimesLastYear, feature_dict['WorkLifeBalance'][WorkLifeBalance],
        YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
    ]

    # Feature names for the input data
    feature_names = [
        "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
        "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
        "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "OverTime", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]

    # Convert the features list to a DataFrame with correct column names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Standardize and predict
    if st.button("Predict"):
        try:
            # Standardize the input features
            standardized_features = scaler.transform(features_df)

            # Perform prediction
            prediction = model.predict(standardized_features)

            # Display prediction
            if prediction[0] == 1:
                st.success("The employee is likely to leave the company.")
            else:
                st.success("The employee is likely to stay with the company.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

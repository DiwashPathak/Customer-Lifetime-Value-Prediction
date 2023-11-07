import json

import numpy as np
import pandas as pd
import streamlit as st

from pipelines.deployment_pipeline import prediction_service_loader

def main():
    st.title("End to End Customer Lifetime Value Prediction Pipeline in ZenML")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict customer lifetime value for the future customers."""
    )

    # st.markdown(
    #     """ 
    # Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    # """
    # )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer lifetime value for a given customer details. You can input the features listed below and get the customer lifetime value. 
    | Variable                      | Description                                                          |
    | ----------------------------- | -------------------------------------------------------------------- |
    | Annual Income                 | Annual income of the person.                                        |
    | Monthly Premium Auto          | The monthly premium amount for auto insurance.                       |
    | Total Payment Amount          | The total amount paid by the customer.                               |
    | Months Since Last Claim       | Number of months since the last insurance claim.                    |
    | Months Since Policy Inception | Number of months since the inception of the insurance policy.       |
    | Open Complaint Count          | The count of open complaints with the insurance company.             |
    | Number of Policies            | The number of insurance policies held by the customer.               |
    | Total Claim Amount            | The total amount claimed by the customer from their insurance policies. |
    | Response                      | Indicates whether the customer responded to an offer or marketing campaign. It has values "Yes" or "No."     |
    | EmploymentStatus              | The employment status of the customer, such as "Employed" or "Unemployed."    |
    | Gender                        | The gender of the customer ie Male or Female.  |
    | Marital Status                | The marital status of the customer, such as "Married" or "Single."     |
    | Policy Type                   | Describes the type of insurance policy, such as "Corporate Auto" or "Personal Auto." |
    | Sales Channel                 | The channel through which the customer purchased or interacted with the insurance company, such as "Agent" or "Call Center." |
    | Vehicle Class                 | Describes the class of the customer's vehicle, e.g., "Two-Door Car," "Four-Door Car," or "SUV." |
    | Insurance Coverage            | Describes the type of insurance coverage the customer has, such as "Basic," "Extended," or "Premium." |
    | Education                     | The highest level of education the customer has completed.           |
    | Vehicle Size                  | The size category of the customer's vehicle, such as "Medsize."     |
    

    """
    )
    number_of_policies = st.sidebar.slider("Number of Policies")
    number_of_open_complaints = st.sidebar.slider("Number of Open Complaints")
    months_since_last_claim = st.sidebar.slider("Months Since Last Claim")
    monthly_premium_auto = st.number_input("Monthly Premium Auto")
    months_since_policy_inception = st.number_input("Months Since Policy Inception")
    total_claim_ammount = st.number_input("Total Claim Amount")
    income = st.number_input("Income($)")
    policy_type = st.selectbox("Policy type", ['Corporate Auto', 'Personal Auto', 'Special Auto'])
    sales_channel = st.selectbox("Sales Channel", ['Agent', 'Call Center', 'Web', 'Branch'])
    vehicle_class = st.selectbox("Vehicle Class", ['Two-Door Car', 'Four-Door Car', 'SUV', 'Luxury SUV', 'Sports Car','Luxury Car'])
    vehicle_size = st.selectbox("Vehicle Size", ['Large', 'Medsize', 'Small'])
    coverage = st.selectbox("Coverage", ['Basic', 'Extended', 'Premium'])
    education = st.selectbox("Education", ['Bachelor', 'College', 'Master', 'High School or Below', 'Doctor'])
    response = st.selectbox("Response", ['Yes', 'No'])
    employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Single'])

    # Encoding input for policy_type
    policy_type_personal_auto = 1 if policy_type == "Personal Auto" else 0
    policy_type_special_auto = 1 if policy_type == "Special Auto" else 0

    # Encoding for responce, employment_status, gender, marital_status
    response = 1 if response == 'Yes' else 0
    employment_status = 1 if employment_status == 'Employed' else 0
    gender = 1 if gender =='Male' else 0
    marital_status = 1 if marital_status == 'Married' else 0

    # Encoding input for sales_channel
    sales_channel_call_center = 1 if sales_channel == 'Call Center' else 0
    sales_channel_web = 1 if sales_channel == 'Web' else 0
    sales_channel_branch = 1 if sales_channel == 'Branch' else 0

    # Encoding input for vehicle_class
    vehicle_class_two_door =1 if vehicle_class ==  'Two-Door Car' else 0
    vehicle_class_suv =1 if vehicle_class ==  'SUV' else 0
    vehicle_class_luxury_suv = 1 if vehicle_class =='Luxury SUV' else 0
    vehicle_class_sports_car =1 if vehicle_class == 'Sports Car' else 0
    vehicle_class_luxury_car =1 if vehicle_class == 'Luxury Car' else 0

    # Encoding input for coverage , education and vehicle size
    coverage = 2 if coverage == 'Premium' else 1 if coverage == 'Extended' else 0
    vehicle_size = 2 if vehicle_size == 'Large' else 1 if vehicle_size == 'Medsize' else 0

    if education == 'Doctor':
        education = 3
    elif education == 'Master':
        education = 2
    elif education == 'Bachelor' or education == 'College':
        education = 1
    elif education == 'High School or Below':
        education = 0

    if st.button('Predict'):
         service = prediction_service_loader(
             pipeline_name = 'continuous_deployment_pipeline',
             pipeline_step_name = 'mlflow_model_deployer_step',
             running = False
         )
         service.start()
         if service is None:
             st.write(
                 'No service could be found.'
             )
         data = {
            "Income": [income],
            "Monthly Premium Auto": [monthly_premium_auto],
            "Months Since Last Claim": [months_since_last_claim],
            "Months Since Policy Inception": [months_since_policy_inception],
            "Number of Open Complaints": [number_of_open_complaints],
            "Number of Policies": [number_of_policies],
            "Total Claim Amount": [total_claim_ammount],
            "Response": [response],
            "EmploymentStatus": [employment_status],
            "Gender": [gender],
            "Marital Status": [marital_status],
            "Policy Type_Personal Auto": [policy_type_personal_auto],
            "Policy Type_Special Auto": [policy_type_special_auto],
            "Sales Channel_Branch": [sales_channel_branch],
            "Sales Channel_Call Center": [sales_channel_call_center],
            "Sales Channel_Web": [sales_channel_web],
            "Vehicle Class_Luxury Car": [vehicle_class_luxury_car],
            "Vehicle Class_Luxury SUV": [vehicle_class_luxury_suv],
            "Vehicle Class_SUV": [vehicle_class_suv],
            "Vehicle Class_Sports Car": [vehicle_class_sports_car],
            "Vehicle Class_Two-Door Car": [vehicle_class_two_door],
            "Coverage": [coverage],
            "Education": [education],
            "Vehicle Size": [vehicle_size],
         }

         df = pd.DataFrame(data)
         json_list =  json.loads(json.dumps(list(df.T.to_dict().values())))
         data = np.array(json_list)
         predection = service.predict(data) 
         st.success(
             f'Predicted customer lifetime value:  {predection}'
         ) 

if __name__ == '__main__':
    main()
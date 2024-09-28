# Churn Model for Predicting Employees Leaving the Company
![image](https://github.com/user-attachments/assets/8cc9c92f-456b-4152-9fd4-6dd37ab46ffd)

## 📚 Table of Contents 📚
[1. Introduction](#section0)<br>
[2. Problem Statement](#section1)<br>
[3. Project Goals](#section1a)<br>
[4. Data Overview](#section1b)<br>
[5. Approach to Problem](#section2)<br>
[6. Libraries Used](#section3)<br>
[7. Data Preprocessing](#section4)<br>
[8. Exploratory Data Analysis](#section5)<br>
[9. Model Development](#section6)<br>
[10. Model Deployment](#section7)<br>

<a id=section0></a>
## 1. Introduction

Employee turnover can be a costly challenge for any organization, leading to disruptions in operations, lower productivity, and the need for frequent recruitment. To mitigate this, a churn model can help predict which employees are most at risk of leaving the company, allowing HR departments to take proactive steps to retain talent. This project explores the development of a churn prediction model using machine learning algorithms, with a focus on identifying key factors influencing employee turnover and guiding the creation of targeted retention programs.

<a id=section1></a>
## 2. Problem Statement

The goal of this project is to build a scalable churn model that predicts which employees are at risk of leaving an organization. The insights from this model help HR departments proactively address employee concerns, improve retention, and target employees most likely to churn. The model was developed using Random Forest algorithms and deployed for real-time predictions.

<a id=section1a></a>
## 3. Project Goals

- **Identify At-Risk Employees**: Use machine learning models to predict which employees may be considering leaving the company.
- **Understand Turnover Drivers**: Analyze data to uncover factors contributing to employee churn, such as job satisfaction, number of projects, working hours, and department.
- **Enhance Retention Strategies**: Use insights from the churn model to inform HR policies and develop programs that improve employee satisfaction and retention.

<a id=section1b></a>
## 4. Data Overview

The dataset used for this project consists of several key employee metrics:
- **Satisfaction Level**: Employee’s self-reported job satisfaction.
- **Last Evaluation**: Performance score from the most recent evaluation.
- **Number of Projects**: Total number of projects the employee has worked on.
- **Average Monthly Hours**: Average number of hours worked per month.
- **Time Spent at Company**: Number of years the employee has worked at the company.
- **Work Accident**: Whether the employee has experienced a work-related accident.
- **Promotion in Last 5 Years**: Whether the employee was promoted in the last five years.
- **Department**: The department the employee works in (e.g., IT, Sales, Technical, etc.).
- **Salary**: The salary level of the employee (Low, Medium, High).

The target variable is **Quit_the_Company**, indicating whether the employee left the company.

<a id=section2></a>
## 5. Approach to Problem
- We aim to build a predictive model to identify employees at risk of leaving the company.
- The dataset was fetched and merged using **Google BigQuery** SQL queries.
- After preprocessing the data, machine learning models such as **Random Forest** and **Gradient Boosting** were used for classification.
- The focus of the model is on **Recall**, as it is essential to identify as many employees at risk of churn as possible.
- The final model was deployed using **PyCaret**.

<a id=section3></a>
## 6. Libraries Used

The project utilized the following key libraries in Python:
- Google BigQuery: Used to connect and query datasets stored in Google Cloud's BigQuery.
- Pandas: Used for data manipulation and analysis, including reading and merging datasets.
- PyCaret: A low-code machine learning library used to train and evaluate multiple machine learning models, including Random Forest, LightGBM, XGBoost, and others.
- Scikit-Learn: For machine learning algorithms and preprocessing, including Random Forest Classifier, Decision Trees, and evaluation metrics like accuracy, precision, recall, and F1 scores.
- SQL (Google BigQuery): SQL queries were used to join datasets from the tbl_hr_data and tbl_new_employees tables.
- Google Colab: Used as the environment for running the notebook and model training.
- Google Looker: For creating an interactive data visualization dashboard, allowing users to explore employee churn trends and filter results by department.

<a id=section4></a>
## 7. Data Preprocessing

- The employee data was stored in two separate tables: `tbl_hr_data` for the original dataset and `tbl_new_employees` for the new employees in the pilot program. These two tables were combined using SQL in **Google BigQuery** with the following query:

```sql
SELECT *, "Original" as Type FROM `data-analysis-end-to-end.employeedata.tbl_hr_data`
UNION ALL
SELECT *, "Pilot" as Type FROM `data-analysis-end-to-end.employeedata.tbl_new_employees`
```

<a id=section5></a>
## 8. Exploratory Data Analysis

Key insights gained from EDA include:
- **Satisfaction Level**: A major predictor of churn. Employees with lower satisfaction scores are more likely to leave.
- **Tenure**: Employees who have been with the company for more than two years or fewer than one year show higher churn rates.
- **Work Accident**: Surprisingly, having a work accident had little effect on churn probability.
- **Department Analysis**: Certain departments, like Support and Technical, have higher churn rates compared to others.

<a id=section6></a>
## 9. Model Development

The Random Forest model was chosen for its ability to handle large datasets and identify feature importance. Below are the steps involved:
- **Feature Engineering**: Created new features from available data, such as `employee engagement` based on the number of projects and working hours.
- **Model Selection**: We compared several models, including Random Forest, LightGBM, and XGBoost, using accuracy, recall, AUC, and F1 score as evaluation metrics.

The following machine learning models were tested, and the best-performing model was **Random Forest**, achieving the highest overall accuracy and performance across multiple metrics:

```
# set up our model
setup(df, target = 'Quit_the_Company', session_id = 123, ignore_features = ['employee_id'], categorical_features = ['salary', 'Departments'])

# false positve, precision ensures that when an employee churns, it is correct. basically being correct in our predictions
# false negative, recall captures most of our employees that will churn
compare_models()
```

<img width="640" alt="Screenshot 2024-09-28 at 6 31 51 PM" src="https://github.com/user-attachments/assets/27e24055-c39d-4318-abf2-43159de41da9">

The best-performing model was the **Random Forest Classifier**, which achieved an accuracy of **98.86%**, an AUC of **0.9912**, and a recall of **95.84%**.

Next, the feature importance plot shows which variables most influenced the model's decision to predict whether an employee will quit. The updated importance ranking is explained in the comments, with satisfaction_level, time_spend_company, and number_project as the key factors.
```
# write back to BigQuery
new_predictions.to_gbq('employeedata.pilot_predictions',
                       project_id,
                       chunksize = None,
                       if_exists = 'replace')

# now, find which variable or column(s) led to our model prediction whether an employee will churn
plot_model(rf_model, plot='feature')
```
![image](https://github.com/user-attachments/assets/fcbf7723-2512-4f97-bc46-537c1bfd20f6)


- X-axis (Variable Importance): Shows the relative importance of each feature in influencing the model’s prediction.
- A higher value means that the feature has more influence on whether an employee is predicted to stay or quit.
- Y-axis (Features): The specific features or variables from the dataset, such as 'satisfaction_level', 'time_spend_company', and 'number_project'.

### Key observations from the updated feature importance are:
1. 'Satisfaction Level' remains the most crucial feature for predicting employee churn. Employees who are less satisfied with their jobs are more likely to quit.
2. 'Time Spent at the Company' is the next most important feature, showing that employees who have been with the company for a longer period may have a higher chance of staying.
3. 'Number of Projects' is another influential factor. Employees who are involved in more projects tend to stay engaged and are less likely to churn.
4. 'Average Monthly Hours' and 'Last Evaluation' also play a moderate role in predicting churn, where higher hours worked and better evaluations are associated with lower churn.
5. Features like 'Work Accident' and salary levels (low, high, medium) have less influence on the model’s predictions, indicating that these factors are less critical in determining employee churn compared to satisfaction and tenure.


<a id=section7></a>
## 10. Model Deployment
The final churn prediction model was deployed using **Google Looker** to create an **interactive dashboard**. The dashboard allows users to:

- Visualize key metrics such as satisfaction levels, time spent at the company, and churn predictions.
- Filter the data by **specific departments** (e.g., Sales, Marketing, IT, etc.) or view results for **all departments** combined.
- Gain insights into the factors driving employee churn through interactive charts, with a focus on **department-level analysis**.

This interactive dashboard provides a user-friendly platform for HR teams to explore employee churn trends and identify actionable insights.

---

**Dashboard Features:**

1. **Interactive Department Filtering**: Users can filter by individual departments or view data across all departments.
2. **Churn Drivers**: Visualizations showing the most important factors contributing to employee churn, such as job satisfaction, time spent at the company, and work accidents.
3. **Employee Retention Metrics**: HR teams can quickly identify the number of employees predicted to stay or leave based on the Random Forest model's output.

You may view the dashboard [_here_](https://lookerstudio.google.com/reporting/fd2a908a-370b-444e-896e-ea5ea94ebf98), or preview a screenshot below:

<img width="433" alt="Screenshot 2024-09-28 at 6 25 49 PM" src="https://github.com/user-attachments/assets/7b9dfc66-5791-4d94-aff9-8dc3271aeef4">

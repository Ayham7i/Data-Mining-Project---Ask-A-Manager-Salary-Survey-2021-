# Data Mining Project Documentation

## 1. Project Overview

**Title:** Data Mining on Ask A Manager Salary Survey 2021**Dataset Source:** [Ask A Manager Salary Survey 2021 (Responses).xlsx](#) (downloaded from the Ask A Manager website)**Objective:**

- Prepare and clean the dataset for further data mining analysis.
- Ensure that the dataset meets the project requirements (at least 10 columns with 2066 records).
- Apply various data mining algorithms for pattern discovery and predictive analysis.

**Dataset Details:**

- **Number of Columns:** 16 (project requirement was a minimum of 10 columns)
- **Number of Records:** 2066

---

## 2. Data Acquisition

The dataset was downloaded from the Ask A Manager website. The file, named **"Ask A Manager Salary Survey 2021 (Responses).xlsx"**, contains salary survey responses from professionals, providing a rich mix of categorical and numerical data that is suitable for various data mining techniques.

---

## 3. Data Cleaning and Preprocessing

Before applying any data mining algorithms, extensive data cleaning and transformation were performed using Microsoft Excel and Kutools. The following steps were taken:

### 3.1. Removing Unnecessary Columns

- **Rationale:**The original dataset contained 16 columns. Irrelevant or redundant columns were removed to focus on essential attributes, ensuring the dataset met the minimum requirement of 10 columns.
- **Outcome:**
  A streamlined dataset that retains only the necessary columns.

### 3.2. Handling Missing Values

#### 3.2.1. Missing Values in Categorical Data

- **Method:**Missing values in categorical fields (e.g., Gender) were imputed using the **mode**.
- **Example Formula (Excel):**
  ```excel
  =INDEX(Q:Q; MODE(IF(Q:Q<>""; MATCH(Q:Q; Q:Q; 0))))
  ```







![Filling missing values for categorical data](image\project_Documentation\1.jpg)

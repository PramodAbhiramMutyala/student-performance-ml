# DIET B.Tech CSE Internship & Project Report Outline

```
========================================================================================
       DHANEKULA INSTITUTE OF ENGINEERING & TECHNOLOGY (DIET), VIJAYAWADA
             DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
                      ACADEMIC YEAR: 2026 – 2027
========================================================================================
INTERNSHIP & MAJOR PROJECT REPORT
Title      : Student Performance Prediction and Academic Risk Analysis Using Machine Learning
Student    : Mutyala Pramod Abhiram
Roll No    : 238T1A05B2
Program    : Bachelor of Technology (B.Tech) in Computer Science & Engineering
Domain     : Artificial Intelligence & Machine Learning (AI & ML)
Duration   : 04 May 2026 to 27 June 2026 (8 Weeks)
Partner Org: ExcelR Edtech Pvt. Ltd., in collaboration with APSCHE
========================================================================================
```

> **Academic Note on Integrity & Institutional Verification:**  
> This comprehensive document provides the full 20–25 page report structure formatted according to Dhanekula Institute of Engineering & Technology guidelines. Independent technical project contributions, empirical ML benchmarks, and system implementations are factually documented from working code. Institutional verification sections, supervisor sign-offs, and verified daily internship logs are demarcated with official institutional placeholders to maintain complete academic integrity.

---

## Preliminary Pages Structure (Pages 1–5)

### Title Page
- Institution Name: Dhanekula Institute of Engineering & Technology (Affiliated to JNTUK)
- Department: Department of Computer Science and Engineering
- Report Title: **Student Performance Prediction and Academic Risk Analysis Using Machine Learning**
- Submitted By: **Mutyala Pramod Abhiram** (Reg. No: `238T1A05B2`)
- Under the guidance of: *[Institutional Guide Name, Designation]*
- Academic Year: 2026–2027

### Certificate of the Institution
- Official declaration by Head of the Department and Internal Project Guide confirming bona fide internship and project submission.
- *[Placeholder: Signature of Internal Guide, Head of Department (CSE), Principal DIET]*

### Certificate from ExcelR Edtech Pvt. Ltd. & APSCHE
- Verification of 8-week internship completion (04 May 2026 – 27 June 2026) in the domain of Artificial Intelligence and Machine Learning.
- *[Placeholder: Official Corporate Certificate ID, Authorized Signatory & Seal]*

### Declaration by the Student
- Formal declaration of independent academic work and adherence to anti-plagiarism standards.

### Acknowledgements
- Grateful acknowledgement to Management, Principal Dr. Ravi Kadiyala, Head of CSE Department, faculty mentors, ExcelR Edtech trainers, and APSCHE coordinators.

### Table of Contents, List of Figures & List of Tables
- Structured page mapping across all 8 chapters.

---

## Chapter 1: Executive Summary & Abstract (Page 6)
- **1.1 Abstract:**
  Early detection of academic underperformance enables timely pedagogical counseling before course completion. Conventional predictive modeling in education suffers from circular target leakage by incorporating late-stage examination marks ($G1$ and $G2$). This project develops an end-to-end educational analytics system utilizing the UCI Student Performance dataset (395 records) that strictly isolates early-warning variables (demographic, family, social habits, attendance, and study hours). Multiple candidate regression pipelines (Dummy baseline, Ridge, RandomForest, HistGradientBoosting) and risk classifiers are benchmarked. The selected RandomForestRegressor achieves a test set MAE of 2.95 and RMSE of 3.95 on the 0–20 scale, while the academic risk classifier achieves a Macro F1-score of 0.55 under balanced class weighting. The model is deployed through a modular FastAPI server, an interactive Bootstrap 5 frontend featuring Chart.js visual analytics, and an audit-trailed SQLite database.
- **1.2 Key Project Highlights & Contributions**

---

## Chapter 2: Organization Overview & Internship Domain (Pages 7–8)
- **2.1 Organization Profile: ExcelR Edtech Pvt. Ltd. in Collaboration with APSCHE:**
  - Corporate background, educational mission, and skill initiatives in Andhra Pradesh.
  - Training curriculum and internship objectives in Artificial Intelligence and Machine Learning.
- **2.2 Internship Schedule & Domain Overview:**
  - Domain: Artificial Intelligence & Machine Learning (Supervised Learning, Model Pipelines, RESTful ML Deployment).
  - Duration: 04 May 2026 to 27 June 2026 (8 Weeks).
- **2.3 Weekly Internship Activity Log Summary:**
  - *Week 1 (04 May – 10 May 2026):* Python environment setup, scikit-learn review, data manipulation with pandas.
  - *Week 2 (11 May – 17 May 2026):* Exploratory Data Analysis, statistical plotting with Matplotlib and Seaborn.
  - *Week 3 (18 May – 24 May 2026):* Supervised learning concepts, linear regression, decision tree mechanics.
  - *Week 4 (25 May – 31 May 2026):* Ensemble techniques (Random Forests, Gradient Boosting), cross-validation.
  - *Week 5 (01 June – 07 June 2026):* Preprocessing pipelines, ColumnTransformer, hyperparameter tuning.
  - *Week 6 (08 June – 14 June 2026):* Backend architecture with FastAPI, Pydantic data validation schemas.
  - *Week 7 (15 June – 21 June 2026):* Database integration with SQLAlchemy/SQLite, frontend dashboard development.
  - *Week 8 (22 June – 27 June 2026):* End-to-end integration, automated testing with pytest, presentation preparation.
  - *(Note: Detailed daily sign-off log sheets to be appended from certified physical internship diary).*

---

## Chapter 3: Problem Definition, Objectives & Scope (Pages 9–10)
- **3.1 Problem Background:**
  - Secondary education attrition and subject failures in foundational mathematics.
  - Limitations of delayed semester-end interventions.
  - Circular data leakage in prior published literature.
- **3.2 Research Questions & Project Objectives:**
  - Can early socio-behavioral attributes predict mathematical performance without intermediate marks?
  - How can predicted continuous scores be converted into actionable pedagogical risk tiers?
  - How to deliver a production-ready, accessible dashboard for non-technical academic mentors?
- **3.3 Scope & Ethical Boundaries:**
  - Advisory decision-support system vs. non-punitive governance.

---

## Chapter 4: System Requirements & Architecture Design (Pages 11–13)
- **4.1 Software & Hardware Requirements:**
  - Python 3.11+, FastAPI, Uvicorn, scikit-learn, SQLAlchemy, SQLite, Bootstrap 5, Chart.js.
  - Standard workstation (8GB RAM, modern multi-core x64 CPU).
- **4.2 Modular Layered Architecture:**
  - Presentation Tier (Bootstrap 5, Jinja2, Vanilla JS, Chart.js).
  - Application Tier (FastAPI RESTful endpoints, Pydantic schemas, dependency injection).
  - Machine Learning Engine (scikit-learn ColumnTransformer & Estimator pipelines).
  - Persistence Tier (SQLite ACID storage with SQLAlchemy ORM).
- **4.3 Architectural Workflow Diagram (Mermaid Representation)**
- **4.4 Database Schema Design:**
  - Table: `prediction_logs` (id, timestamp, prediction_type, model_version, predicted_score, risk_category, record_count, metadata_json).

---

## Chapter 5: Dataset Exploration, Hygiene & Validation (Pages 14–16)
- **5.1 UCI Student Performance Dataset Origin:**
  - Cortez and Silva (2008), 395 secondary students, 33 variables.
- **5.2 Feature Categorization & Variable Dictionary:**
  - 13 Numerical Features (age, parental education, study time, absences, alcohol consumption, family quality).
  - 17 Categorical Features (school, sex, address, family size, support mechanisms, activities, higher education ambitions).
  - Target: $G3$ (0–20 Portuguese secondary mathematics scale).
- **5.3 Exploratory Data Analysis & Empirical Observations:**
  - Grade distribution ($G3$ mean = 10.42, standard deviation = 4.58).
  - Study time correlation (positive relationship between weekly hours & mean score).
  - Absences and past failures (strongest negative predictors of academic success).
- **5.4 Automated Schema Validation:**
  - Multi-tier integrity verification in `src/ml/validation.py`.

---

## Chapter 6: Machine Learning Methodology & Implementation (Pages 17–19)
- **6.1 Early-Warning Feature Isolation:**
  - Strict mathematical exclusion of $G1$, $G2$, and $G3$ from training matrix $X$.
- **6.2 Preprocessing Pipeline:**
  - Missing value imputation (Median for continuous, mode for categorical).
  - Standardization via `StandardScaler`.
  - One-Hot encoding via `OneHotEncoder(handle_unknown='ignore')`.
- **6.3 Model Candidate Benchmarks & Cross-Validation:**
  - Candidate comparison: DummyRegressor, Ridge Regression, RandomForestRegressor, HistGradientBoostingRegressor.
  - 5-fold training cross-validation results vs. hold-out test results.
- **6.4 Academic Risk Classification:**
  - Portuguese educational grading thresholds: High Risk ($G3 < 10$), Moderate Risk ($10 \le G3 < 14$), Low Risk ($G3 \ge 14$).
  - Balancing class weights in RandomForestClassifier.
  - Confusion matrix and macro F1 analysis.
- **6.5 Feature Importance Extraction:**
  - Leading predictors identified: `absences`, `failures`, `Medu`, `Fedu`, `studytime`.

---

## Chapter 7: Web Application Development & User Interface (Pages 20–22)
- **7.1 Page 1: Dashboard (`/`):**
  - KPI metric cards, real-time Chart.js distribution plots, risk doughnut breakdown.
- **7.2 Page 2: Explore Dataset (`/dataset`):**
  - Searchable, paginated record browser, schema dictionary, CSV upload validation tester.
- **7.3 Page 3: Predict Performance (`/predict`):**
  - 30-feature validated Bootstrap form, interactive score progress gauge, risk badge, tailored pedagogical interventions.
- **7.4 Page 4: Batch Prediction (`/batch-predict`):**
  - Drag-and-drop CSV upload, row-level error reporting, enriched CSV downloadable export.
- **7.5 Page 5: Model Evaluation (`/model-evaluation`):**
  - Candidate benchmark tables, hold-out test confusion matrix, feature importance horizontal bar chart.
- **7.6 Page 6: Prediction History (`/history`):**
  - Paginated audit log of prior model inferences, confirmation modal for demonstration reset.

---

## Chapter 8: Verification, Testing & Quality Assurance (Pages 23–24)
- **8.1 Automated Testing Strategy (29 Test Cases):**
  - Test suites: `test_ml.py` (10 tests), `test_api.py` (17 tests), `test_database.py` (2 tests).
  - In-memory database testing fixtures and synthetic student profiles.
- **8.2 API Health, Limit Handling & Security:**
  - File upload restrictions (5 MB max), row limits (5,000 records).
  - Safe CSV stream parsing, code-execution mitigation.
- **8.3 Browser & Route Smoke Testing:**
  - Automated status 200 verification across all 6 Jinja2 HTML routes.

---

## Chapter 9: Conclusion, Limitations & Future Work (Page 25)
- **9.1 Summary of Outcomes:**
  - Successful construction of an end-to-end, reproducible ML application without proprietary framework lock-in.
  - Validated early-warning performance on real student records.
- **9.2 Limitations:**
  - Specificity to Portuguese secondary education curriculum.
  - Modest sample size ($N = 395$).
- **9.3 Future Enhancements:**
  - Multi-institution longitudinal tracking.
  - Natural Language Processing (NLP) integration for qualitative mentor feedback.
  - Migration to PostgreSQL for high-concurrency cloud deployments.
- **9.4 References:**
  - Standard IEEE citations of Cortez & Silva (2008), scikit-learn (Pedregosa et al.), FastAPI documentation, and educational analytics literature.

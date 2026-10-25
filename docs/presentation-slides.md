# DIET B.Tech CSE Viva Presentation Slide Deck (22 Slides)

```
========================================================================================
       DHANEKULA INSTITUTE OF ENGINEERING & TECHNOLOGY (DIET), VIJAYAWADA
             DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
========================================================================================
PROJECT TITLE: Student Performance Prediction and Academic Risk Analysis Using Machine Learning
STUDENT      : Mutyala Pramod Abhiram (Roll No: 238T1A05B2)
ACADEMIC YEAR: 2026 – 2027
INTERNSHIP   : ExcelR Edtech Pvt. Ltd. in collaboration with APSCHE (04 May – 27 June 2026)
========================================================================================
```

---

### Slide 1: Title & Candidacy
- **Heading:** Student Performance Prediction and Academic Risk Analysis Using Machine Learning
- **Sub-heading:** B.Tech Major Project & Internship Technical Viva Presentation
- **Presenter:** Mutyala Pramod Abhiram (`238T1A05B2`)
- **Department:** Computer Science and Engineering, Dhanekula Institute of Engineering & Technology
- **Internship Collaboration:** ExcelR Edtech Pvt. Ltd. & APSCHE (AI & ML Domain)
- **Speaker Notes:**
  > "Respected examiners, Head of the Department, and faculty members, good morning. I am Mutyala Pramod Abhiram, Roll Number 238T1A05B2. Today, I present our engineering project: Student Performance Prediction and Academic Risk Analysis Using Machine Learning, developed as part of our academic work and enriched by technical internship training under ExcelR Edtech and APSCHE."

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Student academic failure is often identified only after end-of-term examination results, when remedial counseling is too late.
- **The Research Gap in Prior Work:** Published educational data mining models often rely heavily on intermediate test marks ($G1$ and $G2$). While this inflates predictive metrics ($R^2 > 0.80$), it creates circular leakage and cannot be used for proactive early warnings at term commencement.
- **Our Solution:** A rigorous early-warning predictive system that strictly isolates early socio-behavioral attributes, estimates final performance, and maps students into actionable risk tiers.
- **Speaker Notes:**
  > "Traditional models cheat by using second-term exam grades to predict final grades. That is not an early warning. Our project addresses this gap by eliminating intermediate grades entirely and using early demographic and study habits to forecast academic risk before exams occur."

---

### Slide 3: Project Objectives
- Acquire, validate, and explore the standardized UCI Student Performance dataset (Mathematics).
- Implement strict early-warning feature isolation by eliminating target leakage.
- Benchmark multiple candidate regression algorithms (Dummy baseline, Ridge, RandomForest, HistGradientBoosting).
- Implement an academic risk classification engine with cost-sensitive class balancing.
- Build an interactive, production-grade web dashboard utilizing FastAPI, Bootstrap 5, and SQLite.
- Maintain comprehensive automated test coverage with pytest.
- **Speaker Notes:**
  > "Our primary objective is not merely training a script in a notebook, but engineering a complete, working software application that is reproducible, rigorously tested, and easily accessible to educators."

---

### Slide 4: Internship Context & Organizational Alignment
- **Partner Organization:** ExcelR Edtech Pvt. Ltd., in collaboration with APSCHE.
- **Internship Domain:** Artificial Intelligence & Machine Learning (04 May 2026 to 27 June 2026).
- **Core Competencies Developed:**
  - Exploratory Data Analysis & statistical visualization.
  - Scikit-learn Pipeline construction & ColumnTransformers.
  - Asynchronous web service development using FastAPI.
  - Database persistence and test-driven development (TDD).
- **Speaker Notes:**
  > "During our 8-week internship with ExcelR Edtech and APSCHE, we received hands-on training in machine learning lifecycle management. This independent project puts those enterprise software engineering principles into practice."

---

### Slide 5: System Architecture Overview
- **Visual:** Block diagram showing 4 distinct tiers:
  - **Presentation Tier:** HTML5, CSS3, Bootstrap 5, Vanilla JS, Chart.js.
  - **Application Tier:** FastAPI, Pydantic v2 schemas, REST endpoints.
  - **Machine Learning Engine:** Scikit-learn Pipelines, joblib serialization.
  - **Data & Audit Tier:** SQLite database (`predictions.db`), raw CSV files, metrics JSON.
- **Key Architectural Decisions:**
  - Lightweight, fast execution with zero bloated JavaScript dependencies.
  - Complete decoupling between inference logic and web presentation.
- **Speaker Notes:**
  > "Here is our four-tier architecture. Notice that the web layer communicates with the machine learning model through clean Pydantic interfaces. The entire application runs natively without heavy external node servers or paid third-party APIs."

---

### Slide 6: Dataset Characteristics & Hygiene
- **Dataset Source:** UCI Machine Learning Repository (Cortez & Silva, 2008).
- **Domain:** Secondary mathematics education (`student-mat.csv`).
- **Scale:** 395 student records, 33 total attributes.
- **Data Quality:** 0 missing values, 0 duplicate records.
- **Variable Groups:**
  - Demographics (school, age, gender, address, family size, parental status).
  - Socio-economic factors (parents' education and employment).
  - Academic habits (travel time, study time, past failures, extracurriculars).
  - Health & lifestyle (alcohol consumption, family relationships, absences).
- **Speaker Notes:**
  > "We utilized the renowned UCI Student Performance dataset. It contains 395 secondary school students evaluated across 30 behavioral and socio-economic variables, providing a rich, multi-faceted ground truth."

---

### Slide 7: Exploratory Data Analysis (EDA) Insights
- **Key Empirical Observations:**
  - **Final Grade ($G3$):** Mean score of 10.42 / 20 with standard deviation of 4.58.
  - **Passing Rate:** 67.09% of students scored $\ge 10.0$; 32.91% were failing.
  - **Zero-Cluster:** 38 students scored exactly 0, reflecting course dropouts or exam absences.
  - **Study Time Correlation:** A clear progressive trend showing higher mean grades as weekly study hours increased from $<2$ hours to $>10$ hours.
  - **Absences Impact:** High absences strongly correlated with failing outcomes.
- **Speaker Notes:**
  > "Our exploratory analysis confirmed two critical educational insights: higher study hours consistently improve average grades, while excessive absences and past failures represent the strongest indicators of academic failure."

---

### Slide 8: Early-Warning Feature Isolation Methodology
- **The Data Leakage Hazard:**
  - $G1$ (First Period Grade) and $G2$ (Second Period Grade) have correlations $>0.85$ with $G3$.
  - Including them makes model predictions trivial but functionally useless at semester onset.
- **Our Strict Isolation Rule:**
  - Feature matrix $X$ is restricted strictly to the 30 non-grade variables.
  - $G1$, $G2$, and $G3$ are dropped before preprocessing.
  - Prevents optimistic evaluation bias and ensures model applicability on day one of school.
- **Speaker Notes:**
  > "This slide illustrates our core scientific contribution: we strictly exclude G1 and G2 from model training. This ensures that the model provides genuine early warnings rather than simply confirming past test scores."

---

### Slide 9: Preprocessing & Scikit-Learn Pipelines
- **Automated ColumnTransformer:**
  - **Numerical Features (13 variables):** Median imputation (`SimpleImputer`) + Z-score normalization (`StandardScaler`).
  - **Categorical Features (17 variables):** Mode imputation + One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`).
- **Benefits:**
  - Encapsulates transformation parameters to eliminate data leakage during cross-validation.
  - Produces a single, self-contained pipeline object serialized via `joblib`.
- **Speaker Notes:**
  > "All preprocessing is packaged inside a scikit-learn ColumnTransformer within a Pipeline. This ensures that imputation statistics and scaling parameters are learned strictly from the training folds, eliminating validation leakage."

---

### Slide 10: Regression Algorithm Benchmark
- **Candidate Comparison (5-Fold Cross-Validation on 80% Train, Evaluated on 20% Test):**
  | Algorithm | CV RMSE | Test MAE | Test RMSE | Test $R^2$ |
  | :--- | :---: | :---: | :---: | :---: |
  | **DummyRegressor (Baseline)** | 4.6384 | 3.2730 | 4.3610 | -0.0004 |
  | **Ridge Regression** | 4.4802 | 3.2278 | 4.3902 | -0.0139 |
  | **RandomForestRegressor (Selected)** | **3.9381** | **2.9471** | **3.9518** | **0.1785** |
  | **HistGradientBoostingRegressor** | 4.0569 | 3.0186 | 4.0507 | 0.1368 |
- **Selection Decision:** RandomForestRegressor achieved the lowest cross-validation RMSE (3.9381) and highest test $R^2$ (0.1785).
- **Speaker Notes:**
  > "Here are our actual empirical test results. Notice that the RandomForestRegressor decisively outperformed both the linear baseline and gradient boosting, achieving an MAE of 2.94 grade points on the test set."

---

### Slide 11: Academic Risk Classification & Class Balancing
- **Mapping Continuous Grades to Actionable Categories:**
  - **High Risk ($G3 < 10.0$):** Failing status &bull; Urgent intervention required.
  - **Moderate Risk ($10.0 \le G3 < 14.0$):** Satisfactory &bull; Regular progress monitoring.
  - **Low Risk ($G3 \ge 14.0$):** Superior performance &bull; Enrichment & peer mentoring.
- **Handling Class Imbalance:**
  - Applied `class_weight='balanced'` in RandomForestClassifier.
  - Results on hold-out test set: **Test Accuracy: 54.43%**, **Macro F1-Score: 0.5494**, **Weighted F1: 0.5354** (vs. Dummy baseline Macro F1 of 0.1964).
- **Speaker Notes:**
  > "In addition to predicting continuous marks, we categorized grades into Portuguese academic risk tiers. Using balanced class weighting, our Random Forest classifier achieved a Macro F1 score of 0.55, vastly outperforming the baseline."

---

### Slide 12: Hold-Out Test Confusion Matrix Analysis
- **Confusion Matrix Breakdown (Hold-Out Test Set, $N = 79$):**
  - High Risk: Successfully flags at-risk students for early counseling.
  - Moderate Risk: Accurately identifies border-line students.
  - Low Risk: Demonstrates high specificity for high achievers.
- **Pedagogical Impact:** False negatives (failing students marked as low risk) are minimized by cost-sensitive weighting.
- **Speaker Notes:**
  > "This confusion matrix demonstrates the real-world utility of the classifier. By weighting the classes appropriately, we minimize dangerous false negatives, ensuring that struggling students are not overlooked."

---

### Slide 13: Feature Importance Analysis
- **Top 5 Predictive Indicators (Extracted from Random Forest Pipeline):**
  1. `absences` — Number of missed classes (Primary predictor of disengagement).
  2. `failures` — Past course failures (Indicator of cumulative foundational learning deficit).
  3. `Medu` / `Fedu` — Parental education levels (Socio-academic home support).
  4. `studytime` — Weekly self-study hours outside the classroom.
  5. `age` — Student age relative to grade cohort.
- **Speaker Notes:**
  > "Inspecting our model's Gini feature importances reveals that attendance and past failure history are the most influential factors, followed closely by parental education and weekly study commitment."

---

### Slide 14: FastAPI Backend Architecture
- **Core Server Capabilities:**
  - Asynchronous request handling with Uvicorn.
  - Pydantic v2 data models with automatic bounds checking.
  - Thread-safe model predictor singleton with memory caching.
  - Safe CSV stream processing with row-level validation.
  - RESTful endpoints: `/health`, `/api/predict`, `/api/predict/batch`, `/api/dataset/predict`, `/api/predictions`.
- **Speaker Notes:**
  > "Our backend is built on FastAPI. It provides sub-millisecond validation using Pydantic, exposes modular REST endpoints, and safeguards system stability with upload size and row-count limits."

---

### Slide 15: Database Persistence & Audit Logging
- **Database Engine:** SQLite accessed via SQLAlchemy ORM.
- **Audit Table (`prediction_logs`):**
  - `id`: Auto-incrementing primary key.
  - `timestamp`: UTC execution timestamp.
  - `prediction_type`: "single", "batch_csv", or "batch_json".
  - `predicted_score` & `risk_category`.
  - `metadata_json`: Non-identifying summary of key parameters.
- **Privacy Compliance:** No private student names or PII are stored.
- **Speaker Notes:**
  > "Every model inference is logged to an SQLite audit table with minimal non-identifying metadata. This creates a transparent audit trail while rigorously protecting student privacy."

---

### Slide 16: UI/UX Design System (Bootstrap 5 & Vanilla JS)
- **Executive Design Palette:** Deep navy (`#0f172a`), slate (`#1e293b`), and vibrant teal (`#0d9488`).
- **Key UX Principles:**
  - Clean cards, accessible typography (Google Fonts Outfit & Inter).
  - Responsive layouts optimized for desktops, tablets, and mobile screens.
  - Zero bloated external JS frameworks (pure Vanilla JS + native Fetch API).
  - Dynamic visual feedback: real-time score gauges, status badges, and toast alerts.
- **Speaker Notes:**
  > "For the user interface, we rejected generic templates in favor of a modern, responsive design system. The dashboard uses crisp typography and tailored color tokens to provide a clean user experience."

---

### Slide 17: Dashboard & Explore Dataset Pages
- **Dashboard (`/`):**
  - KPI summary cards (Dataset records, mean grade, test MAE, test $R^2$).
  - Dynamic Chart.js distributions (grade histograms, study time vs grade, absence impact).
- **Explore Dataset (`/dataset`):**
  - Searchable, filterable 395-student record table.
  - Interactive CSV validation modal testing external files against schema requirements.
- **Speaker Notes:**
  > "Here are the first two pages. The executive dashboard displays real-time charts populated by our analytics API, while the dataset explorer allows educators to search records and validate external CSV schemas."

---

### Slide 18: Individual & Batch Prediction Walkthrough
- **Predict Performance (`/predict`):**
  - 30-feature validated form with instant demo presets (At-Risk, Average, High-Achiever).
  - Output card: Clamped grade score (0–20), risk badge, progress gauge, and tailored interventions.
- **Batch Prediction (`/batch-predict`):**
  - Drag-and-drop CSV upload with validation diagnostics.
  - Row-level error reporting (no silent discards).
  - One-click export of enriched CSV files with predicted grades and risk tiers appended.
- **Speaker Notes:**
  > "Pages three and four handle prediction. On the predict page, clicking a preset immediately shows the estimated score and tailored recommendations. On the batch page, educators can upload an entire class list and export an enriched CSV."

---

### Slide 19: Model Diagnostics & History Pages
- **Model Evaluation (`/model-evaluation`):**
  - Live candidate benchmark tables loaded from saved `metrics.json`.
  - Hold-out confusion matrix and interactive feature importance horizontal bar chart.
- **Prediction History (`/history`):**
  - Paginated audit log with server-side pagination.
  - Safe deletion modal requiring explicit user confirmation to clear demonstration logs.
- **Speaker Notes:**
  > "The model evaluation screen presents live diagnostic metrics and confusion matrices directly from our training artifacts, while the history page maintains a paginated audit trail of all prior inferences."

---

### Slide 20: Comprehensive Automated Testing (pytest)
- **Test Strategy:** 29 automated test cases executed via pytest in under 1 second.
- **Coverage Breakdown:**
  - **ML Suite (10 tests):** Schema validation, ColumnTransformer, feature isolation, single and batch inference.
  - **API Suite (17 tests):** Health check, metrics, predict, batch predict, CSV validation, enriched download, and 6 HTML page smoke tests.
  - **Database Suite (2 tests):** ORM CRUD operations, pagination offset calculations, and table deletion.
- **Speaker Notes:**
  > "To ensure enterprise-grade reliability, we built a comprehensive test suite of 29 automated tests. All unit, integration, and UI smoke tests execute in under a second and pass with 100% success."

---

### Slide 21: Ethical Governance & Responsible AI
- **Advisory Role Only:** Algorithmic estimates must never replace human pedagogical judgment or justify punitive disciplinary actions.
- **Data Protection:** No PII stored; adherence to international educational data guidelines.
- **Action-Oriented:** Predictions trigger early remedial assistance, not tracking or exclusion.
- **Speaker Notes:**
  > "Ethical AI is paramount in education. Our system is explicitly designed as an advisory decision-support tool to connect struggling students with mentors, not as an automated grading or tracking mechanism."

---

### Slide 22: Conclusion & Technical Demonstration
- **Summary of Engineering Achievements:**
  - Built an end-to-end, zero-leakage ML pipeline on real secondary student data.
  - Validated multiple regression and classification algorithms with real test-set metrics.
  - Deployed a full-stack, responsive web application using FastAPI, Bootstrap 5, and SQLite.
  - Successfully verified with 29 automated pytest tests.
- **Thank You!**
  - Open for questions and live system demonstration.
- **Speaker Notes:**
  > "In conclusion, this project demonstrates a complete, reliable, and ethically sound educational analytics application. Thank you, and I am now ready for your questions and the live demonstration."

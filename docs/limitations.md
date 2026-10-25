# Limitations, Ethical Considerations & Responsible AI Statement

## 1. Algorithmic and Data Limitations

### 1.1 Cohort Specificity & Socio-Cultural Context
The underlying predictive models were trained on the UCI Student Performance dataset (Cortez & Silva, 2008), comprising 395 students across two public secondary schools in the Alentejo region of Portugal. Consequently:
- **Demographic Bias:** Cultural and institutional factors (such as the 0–20 grading scale, mandatory secondary curriculum structures, and local family support patterns) do not directly transfer to other international education systems (such as the Indian CBSE/State board system or US letter grades) without dataset re-collection and domain recalibration.
- **Sample Size Constraints:** With 395 total observations, training partitions represent approximately 316 students. While adequate for benchmarking standard algorithms and regularization, deep neural architectures or high-dimensional interactions risk overfitting.

### 1.2 Early-Warning Information Trade-Off
To guarantee a genuine early-warning system, intermediate examination marks (`G1` and `G2`) were strictly excluded from model feature inputs. 
- While this prevents circular leakage, it lowers the explained variance ($R^2 \approx 0.18$, $\text{MAE} \approx 2.95$) compared to models that peek at $G2$ ($R^2 > 0.80$).
- Socio-demographic variables alone inherently capture only a fraction of student academic potential. Factors such as day-to-day effort, illness on exam day, and personal motivation remain unmeasured.

---

## 2. Ethical Considerations & Responsible Use Statement

### 2.1 Non-Punitive, Decision-Support Principle
- **Strictly Advisory:** Model predictions represent statistical estimations and probabilistic risk tiers. Under no circumstances should automated model outputs be used to assign official academic grades, deny course admission, track students into lower academic streams, or administer punitive disciplinary actions.
- **Early Support Orientation:** The sole legitimate purpose of this application is early educational counseling—enabling teachers and academic mentors to proactively allocate extra tutoring, schedule progress reviews, and support students before failures occur.

### 2.2 Human-in-the-Loop Governance
- Predictive results must always be reviewed by qualified human educators who understand the full context of the student's personal circumstances.
- Automated algorithmic assessments must never replace empathetic pedagogical discretion.

---

## 3. Technical & Architectural Limitations

### 3.1 SQLite Deployment Scope
- **Concurrency:** SQLite uses file-level locking for write transactions. While ideal for desktop demonstrations, single-server development, and local academic presentations, multi-instance horizontal scaling in production requires migrating `DATABASE_URL` to PostgreSQL.
- **Backup & Auditing:** Production environments should implement automated snapshot backups and encryption at rest.

### 3.2 Model Invalidation & Concept Drift
- Student populations and curriculum standards evolve over time. Model pipelines must be periodically re-evaluated and retrained using fresh institutional data to prevent performance degradation over successive academic years.

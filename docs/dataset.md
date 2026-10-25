# Dataset Specifications & Data Hygiene

## 1. Overview and Citation
The primary dataset utilized in this project is the **Student Performance Data Set**, collected by Paulo Cortez and Alice Silva (2008) from two Portuguese secondary schools during the 2005–2006 academic year.

- **Repository:** [UCI Machine Learning Repository (Dataset ID: 320)](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Subject Area:** Secondary Mathematics (`student-mat.csv`)
- **Number of Observations:** 395 students
- **Total Variables:** 33 attributes (30 demographic, social, and academic indicators; 2 intermediate period grades; 1 final grade)

**Citation:**
> Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal. EUROSIS, ISBN 978-9077381-39-7.

---

## 2. Variable Dictionary & Schema Specifications

### 2.1 Demographic and Family Attributes
| Variable | Type | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `school` | Categorical | `["GP", "MS"]` | Student's school: Gabriel Pereira or Mousinho da Silveira |
| `sex` | Categorical | `["F", "M"]` | Student's gender |
| `age` | Numerical | `15` to `22` | Student age in years |
| `address` | Categorical | `["U", "R"]` | Home address type: Urban or Rural |
| `famsize` | Categorical | `["LE3", "GT3"]` | Family size: &le; 3 or &gt; 3 members |
| `Pstatus` | Categorical | `["T", "A"]` | Parent's cohabitation status: Together or Apart |
| `Medu` | Numerical | `0` to `4` | Mother's education (0: none, 1: primary, 2: 5-9th, 3: secondary, 4: higher) |
| `Fedu` | Numerical | `0` to `4` | Father's education (0: none, 1: primary, 2: 5-9th, 3: secondary, 4: higher) |
| `Mjob` | Categorical | `["teacher", "health", "services", "at_home", "other"]` | Mother's occupation |
| `Fjob` | Categorical | `["teacher", "health", "services", "at_home", "other"]` | Father's occupation |
| `reason` | Categorical | `["home", "reputation", "course", "other"]` | Reason to choose school |
| `guardian` | Categorical | `["mother", "father", "other"]` | Student's legal guardian |

### 2.2 Academic Environment & Study Habits
| Variable | Type | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `traveltime` | Numerical | `1` to `4` | Travel time (1: &lt;15m, 2: 15-30m, 3: 30-60m, 4: &gt;1h) |
| `studytime` | Numerical | `1` to `4` | Weekly study time (1: &lt;2h, 2: 2-5h, 3: 5-10h, 4: &gt;10h) |
| `failures` | Numerical | `0` to `4` | Past class failures (0 to 4) |
| `schoolsup` | Categorical | `["yes", "no"]` | Extra educational school support |
| `famsup` | Categorical | `["yes", "no"]` | Family educational support |
| `paid` | Categorical | `["yes", "no"]` | Extra paid classes in math |
| `activities` | Categorical | `["yes", "no"]` | Extracurricular activities |
| `nursery` | Categorical | `["yes", "no"]` | Attended nursery school |
| `higher` | Categorical | `["yes", "no"]` | Desires higher education |
| `internet` | Categorical | `["yes", "no"]` | Internet access at home |
| `romantic` | Categorical | `["yes", "no"]` | In a romantic relationship |

### 2.3 Social Lifestyle, Well-Being & Health
| Variable | Type | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `famrel` | Numerical | `1` to `5` | Quality of family relationships (1: very bad to 5: excellent) |
| `freetime` | Numerical | `1` to `5` | Free time after school (1: very low to 5: very high) |
| `goout` | Numerical | `1` to `5` | Going out with friends (1: very low to 5: very high) |
| `Dalc` | Numerical | `1` to `5` | Workday alcohol consumption (1: very low to 5: very high) |
| `Walc` | Numerical | `1` to `5` | Weekend alcohol consumption (1: very low to 5: very high) |
| `health` | Numerical | `1` to `5` | Current health status (1: very bad to 5: very good) |
| `absences` | Numerical | `0` to `93` | Number of school absences |

### 2.4 Academic Grade Targets
| Variable | Role | Range | Purpose |
| :--- | :--- | :--- | :--- |
| `G1` | Intermediate Grade | `0` to `20` | First period grade (**Excluded from early-warning features**) |
| `G2` | Intermediate Grade | `0` to `20` | Second period grade (**Excluded from early-warning features**) |
| `G3` | **Regression Target** | `0` to `20` | Final mathematics grade |
| `Risk` | **Classification Target** | `[High, Moderate, Low]` | Derived academic risk tier |

---

## 3. Data Hygiene & Compliance

### 3.1 Privacy and Anonymity
- The dataset does not contain student names, identification numbers, addresses, IP addresses, or contact information.
- Only non-identifying aggregate attributes are persisted into the local demonstration database during inference auditing.

### 3.2 Data Integrity Verification
Automated validation (`src/ml/validation.py`) enforces:
1. Column completeness: All 30 feature columns must be present.
2. Type checking: Numerical fields must be convertible to floats/integers without loss.
3. Category verification: Categorical values must strictly match defined domain sets.
4. Range checking: Numerical attributes must lie strictly within defined minimum and maximum boundaries.

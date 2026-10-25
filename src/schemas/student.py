"""
Pydantic schemas for student feature inputs.
Enforces rigorous validation across demographic, social, and academic variables.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StudentFeatureInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "school": "GP",
                "sex": "F",
                "age": 17,
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "T",
                "Medu": 3,
                "Fedu": 3,
                "Mjob": "services",
                "Fjob": "other",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 1,
                "studytime": 3,
                "failures": 0,
                "schoolsup": "no",
                "famsup": "yes",
                "paid": "no",
                "activities": "yes",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes",
                "romantic": "no",
                "famrel": 4,
                "freetime": 3,
                "goout": 2,
                "Dalc": 1,
                "Walc": 1,
                "health": 5,
                "absences": 2,
            }
        }
    )
    """
    Schema representing the 30 early-warning indicators for an individual student.
    Strictly excludes G1, G2, and G3 to prevent data leakage.
    """
    # Demographic / Family
    school: Literal["GP", "MS"] = Field(
        default="GP", description="Student school: GP (Gabriel Pereira) or MS (Mousinho da Silveira)"
    )
    sex: Literal["F", "M"] = Field(
        default="F", description="Student sex: F (female) or M (male)"
    )
    age: int = Field(
        default=17, ge=15, le=22, description="Student age between 15 and 22"
    )
    address: Literal["U", "R"] = Field(
        default="U", description="Home address type: U (urban) or R (rural)"
    )
    famsize: Literal["LE3", "GT3"] = Field(
        default="GT3", description="Family size: LE3 (<=3 members) or GT3 (>3 members)"
    )
    Pstatus: Literal["T", "A"] = Field(
        default="T", description="Parent cohabitation status: T (together) or A (apart)"
    )
    Medu: int = Field(
        default=2, ge=0, le=4, description="Mother education (0: none, 1: 4th grade, 2: 5-9th grade, 3: secondary, 4: higher)"
    )
    Fedu: int = Field(
        default=2, ge=0, le=4, description="Father education (0: none, 1: 4th grade, 2: 5-9th grade, 3: secondary, 4: higher)"
    )
    Mjob: Literal["teacher", "health", "services", "at_home", "other"] = Field(
        default="other", description="Mother job"
    )
    Fjob: Literal["teacher", "health", "services", "at_home", "other"] = Field(
        default="other", description="Father job"
    )
    reason: Literal["home", "reputation", "course", "other"] = Field(
        default="course", description="Reason to choose this school"
    )
    guardian: Literal["mother", "father", "other"] = Field(
        default="mother", description="Student guardian"
    )

    # Academic & School Environment
    traveltime: int = Field(
        default=1, ge=1, le=4, description="Home to school travel time (1: <15 min, 2: 15-30 min, 3: 30-60 min, 4: >60 min)"
    )
    studytime: int = Field(
        default=2, ge=1, le=4, description="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)"
    )
    failures: int = Field(
        default=0, ge=0, le=4, description="Number of past class failures (0 to 4)"
    )
    schoolsup: Literal["yes", "no"] = Field(
        default="no", description="Extra educational support from school"
    )
    famsup: Literal["yes", "no"] = Field(
        default="no", description="Family educational support"
    )
    paid: Literal["yes", "no"] = Field(
        default="no", description="Extra paid classes in math"
    )
    activities: Literal["yes", "no"] = Field(
        default="no", description="Extra-curricular activities"
    )
    nursery: Literal["yes", "no"] = Field(
        default="yes", description="Attended nursery school"
    )
    higher: Literal["yes", "no"] = Field(
        default="yes", description="Desires higher education"
    )
    internet: Literal["yes", "no"] = Field(
        default="yes", description="Internet access at home"
    )
    romantic: Literal["yes", "no"] = Field(
        default="no", description="In a romantic relationship"
    )

    # Social, Lifestyle & Health
    famrel: int = Field(
        default=4, ge=1, le=5, description="Quality of family relationships (1: very bad to 5: excellent)"
    )
    freetime: int = Field(
        default=3, ge=1, le=5, description="Free time after school (1: very low to 5: very high)"
    )
    goout: int = Field(
        default=3, ge=1, le=5, description="Going out with friends (1: very low to 5: very high)"
    )
    Dalc: int = Field(
        default=1, ge=1, le=5, description="Workday alcohol consumption (1: very low to 5: very high)"
    )
    Walc: int = Field(
        default=1, ge=1, le=5, description="Weekend alcohol consumption (1: very low to 5: very high)"
    )
    health: int = Field(
        default=5, ge=1, le=5, description="Current health status (1: very bad to 5: very good)"
    )
    absences: int = Field(
        default=4, ge=0, le=93, description="Number of school absences (0 to 93)"
    )

"""
Data Generator for Community Learning Center (CLC) Learning Analytics
Generates realistic non-formal education data including:
- Learner Demographics (anonymized IDs, age group, gender, enrolled programme, prior education)
- Attendance Metrics (attendance rate %, consecutive absences, weekend turnout)
- Academic Progress (baseline score, midterm score, final/recent quiz, homework completion %)
- Engagement Metrics (volunteer interaction hours, participation level, feedback rating)
- Ground truth risk status for training & validation
"""

import numpy as np
import pandas as pd
import os

def generate_clc_dataset(num_learners=250, seed=42):
    np.random.seed(seed)
    
    programmes = [
        "Remedial School Tuition", 
        "Digital Literacy & Computers", 
        "Vocational & Craft Skills", 
        "Adult Continuing Education"
    ]
    programme_weights = [0.40, 0.30, 0.15, 0.15]
    
    learner_ids = [f"CLC-L{1001 + i}" for i in range(num_learners)]
    progs = np.random.choice(programmes, size=num_learners, p=programme_weights)
    genders = np.random.choice(["Female", "Male", "Non-Binary / Other"], size=num_learners, p=[0.55, 0.42, 0.03])
    
    # Age groups aligned with programs
    age_groups = []
    for p in progs:
        if p == "Remedial School Tuition":
            age_groups.append(np.random.choice(["Children (8-12)", "Adolescents (13-17)"], p=[0.6, 0.4]))
        elif p == "Digital Literacy & Computers":
            age_groups.append(np.random.choice(["Adolescents (13-17)", "Young Adults (18-25)", "Adults (26+)"], p=[0.3, 0.5, 0.2]))
        elif p == "Vocational & Craft Skills":
            age_groups.append(np.random.choice(["Young Adults (18-25)", "Adults (26+)"], p=[0.4, 0.6]))
        else: # Adult Continuing Education
            age_groups.append(np.random.choice(["Adults (26-40)", "Senior Adults (41+)"], p=[0.7, 0.3]))
            
    # Academic & Engagement Features
    attendance_rate = []
    consecutive_absences = []
    baseline_score = []
    recent_quiz_score = []
    homework_completion = []
    volunteer_support_hours = []
    participation_score = []
    
    for i in range(num_learners):
        # Latent engagement factor (-2 to +2)
        latent_factor = np.random.normal(0, 1)
        
        # Attendance: mean 72%, modulated by latent engagement
        att = np.clip(int(72 + latent_factor * 18 + np.random.normal(0, 8)), 15, 100)
        attendance_rate.append(att)
        
        # Consecutive absences in last month: higher if attendance is low
        cons_abs = max(0, int(np.clip((100 - att) / 12 + np.random.normal(0, 1.5), 0, 14)))
        consecutive_absences.append(cons_abs)
        
        # Baseline score (0 - 100)
        base = np.clip(int(50 + np.random.normal(0, 15)), 20, 95)
        baseline_score.append(base)
        
        # Progress: driven by attendance and latent enthusiasm
        score_gain = (att - 50) * 0.3 + latent_factor * 5 + np.random.normal(0, 6)
        recent_score = np.clip(int(base + score_gain), 10, 100)
        recent_quiz_score.append(recent_score)
        
        # Homework / Practice completion (0 - 100%)
        hw = np.clip(int(att * 0.85 + np.random.normal(0, 10)), 5, 100)
        homework_completion.append(hw)
        
        # Volunteer 1-on-1 tutoring hours in past 8 weeks
        vol_hrs = round(max(0.5, np.random.exponential(scale=3.5) + (2 if att < 60 else 1)), 1)
        volunteer_support_hours.append(vol_hrs)
        
        # Facilitator-rated participation score (1 - 5 stars)
        part = np.clip(int(np.round(1 + (att / 100.0) * 2.5 + (recent_score / 100.0) * 1.5 + np.random.normal(0, 0.4))), 1, 5)
        participation_score.append(part)
        
    df = pd.DataFrame({
        "learner_id": learner_ids,
        "enrolled_programme": progs,
        "gender": genders,
        "age_group": age_groups,
        "attendance_rate_pct": attendance_rate,
        "consecutive_absences": consecutive_absences,
        "baseline_score": baseline_score,
        "recent_quiz_score": recent_quiz_score,
        "score_improvement": np.array(recent_quiz_score) - np.array(baseline_score),
        "homework_completion_pct": homework_completion,
        "volunteer_hours": volunteer_support_hours,
        "participation_rating_1to5": participation_score
    })
    
    # Calculate Risk Score (Rule-based ground truth + noise to train ML)
    # Risk indicators: Low attendance, high consecutive absence, failing scores, low homework
    risk_points = (
        (df["attendance_rate_pct"] < 60).astype(int) * 35 +
        (df["attendance_rate_pct"].between(60, 75)).astype(int) * 15 +
        (df["consecutive_absences"] >= 5).astype(int) * 25 +
        (df["recent_quiz_score"] < 45).astype(int) * 25 +
        (df["homework_completion_pct"] < 50).astype(int) * 15 +
        (df["score_improvement"] < -5).astype(int) * 10
    )
    
    # Classify Risk Level
    risk_labels = []
    risk_probability = np.clip(risk_points / 100.0 + np.random.normal(0, 0.05, size=num_learners), 0.02, 0.98)
    for p in risk_probability:
        if p >= 0.55:
            risk_labels.append("High Risk")
        elif p >= 0.30:
            risk_labels.append("Moderate Risk")
        else:
            risk_labels.append("Low Risk")
            
    df["risk_level"] = risk_labels
    df["risk_probability"] = np.round(risk_probability, 3)
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(out_dir, "clc_learner_data.csv")
    df = generate_clc_dataset(num_learners=280)
    df.to_csv(data_path, index=False)
    print(f"Dataset generated with {len(df)} records at {data_path}")
    print(df["risk_level"].value_counts())

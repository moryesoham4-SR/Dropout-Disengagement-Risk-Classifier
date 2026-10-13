"""
Dataset generator incorporating the exact Google Form survey questions
for Community Learning Center (CLC) Learning Analytics.

Survey Sections & Fields:
1. Demographics:
   - age_group: Below 15, 15-18, 19-22, Above 22
   - education_level: Primary, Secondary, Higher Secondary, Undergraduate, Other
   - gender: Male, Female, Other
   - occupation_status: Student, Working, Self-employed, Not currently working

2. Attendance and Learning Participation:
   - center_attendance_freq: 3-4 times a week, 1-2 times a week, Rarely
   - attendance_regularity: Very regular, Regular, Sometimes regular, Irregular
   - classroom_participation: Very actively, Actively, Sometimes, Rarely
   - assignment_completion: Always, Often, Sometimes, Never
   - reason_miss_classes: Multi-select: Personal reasons, Health issues, Family responsibilities, Transport problems, Lack of interest

3. Learning Experience and Performance:
   - topic_understanding_ease: Very easy, Easy, Moderate, Difficult, Very Difficult
   - teaching_method_rating: Excellent, Good, Average, Poor, Very poor
   - teacher_help_frequency: Always, Often, Sometimes, Rarely, Never
   - performance_improvement: Improved significantly, Improved, Slightly improved, No improvement, Not sure
   - confidence_level: Very confident, Confident, Somewhat confident, Not confident at all

4. Learning Resources and Technology:
   - resources_used: Multi-select (Textbooks, Notes/Study Material, Library, Computers, Internet Resources, Videos/Online Courses)
   - digital_device_frequency: Daily, Several times a week, Once a week, Rarely, Never
   - primary_device: Smartphone, Laptop, Tablet, Desktop, None
   - resource_availability_rating: 1 to 5
   - tech_comfort_rating: 1 to 5
   - digital_content_type: Multi-select (Educational videos, Online quizes, E-books/PDF's, Learning websites, Mobile apps, Online classes)

5. Learning Challenges and Support:
   - challenges_faced: Multi-select (Difficulty understanding topics, Lack of study time, Lack of learning materials, Internet/technology problems, Language difficulties, Personal/family responsibilities, None)
   - difficulty_frequency: Very often, Often, Sometimes, Rarely, Never
   - asking_teacher_comfort: 1 to 5
   - support_approach: Teacher, Friends/classmates, Family members, Online Resources, I usually study on my own
   - suggestions_text: Qualitative feedback

Target Labels:
- risk_level: Low Risk, Moderate Risk, High Risk (for Early Warning Dropout/Disengagement Classifier)
- learner_cluster: Cluster assignment from K-Means
"""

import numpy as np
import pandas as pd
import os

def generate_survey_dataset(n_samples=320, random_state=42):
    np.random.seed(random_state)
    
    learner_ids = [f"CLC-2026-{1001 + i}" for i in range(n_samples)]
    
    age_groups = np.random.choice(["Below 15", "15-18", "19-22", "Above 22"], size=n_samples, p=[0.25, 0.40, 0.22, 0.13])
    
    edu_levels = []
    occupations = []
    for age in age_groups:
        if age == "Below 15":
            edu_levels.append(np.random.choice(["Primary", "Secondary"], p=[0.45, 0.55]))
            occupations.append("Student")
        elif age == "15-18":
            edu_levels.append(np.random.choice(["Secondary", "Higher Secondary"], p=[0.5, 0.5]))
            occupations.append(np.random.choice(["Student", "Not currently working"], p=[0.9, 0.1]))
        elif age == "19-22":
            edu_levels.append(np.random.choice(["Higher Secondary", "Undergraduate", "Other"], p=[0.25, 0.65, 0.10]))
            occupations.append(np.random.choice(["Student", "Working", "Self-employed", "Not currently working"], p=[0.6, 0.2, 0.1, 0.1]))
        else: # Above 22
            edu_levels.append(np.random.choice(["Secondary", "Higher Secondary", "Undergraduate", "Other"], p=[0.2, 0.2, 0.4, 0.2]))
            occupations.append(np.random.choice(["Working", "Self-employed", "Not currently working"], p=[0.5, 0.3, 0.2]))
            
    genders = np.random.choice(["Female", "Male", "Other"], size=n_samples, p=[0.54, 0.44, 0.02])
    
    # Latent Engagement & Performance Index (-3 to +3)
    engagement_latent = np.random.normal(0, 1.2, size=n_samples)
    
    att_freq_list = []
    att_reg_list = []
    part_list = []
    assign_list = []
    topic_und_list = []
    teach_rating_list = []
    teach_help_list = []
    improvement_list = []
    confidence_list = []
    tech_freq_list = []
    primary_device_list = []
    res_avail_list = []
    tech_comfort_list = []
    diff_freq_list = []
    ask_comfort_list = []
    approach_list = []
    reasons_miss_list = []
    challenges_list = []
    resources_used_list = []
    digital_content_list = []
    suggestions_list = []
    
    suggestion_pool = [
        "I would like more practical activities during classes.",
        "Need extra doubt-clearing sessions on weekends.",
        "Provide more computers for practice as available ones are often occupied.",
        "Internet connection is sometimes slow during online exercises.",
        "Teachers are very helpful and patient, need more study notes in regional language.",
        "Class timings clash with part-time work or domestic chores.",
        "More mock quizzes and assessments to check preparation.",
        "Everything is good, happy with the center.",
        "Need guidance on career opportunities and vocational skill certificates."
    ]
    
    for i in range(n_samples):
        lat = engagement_latent[i]
        
        # 1. Attendance Frequency
        p_att = [0.55 + 0.15 * np.tanh(lat), 0.35 - 0.05 * np.tanh(lat), 0.10 - 0.10 * np.tanh(lat)]
        p_att = np.clip(p_att, 0.02, 0.96)
        p_att /= np.sum(p_att)
        att_freq = np.random.choice(["3-4 times a week", "1-2 times a week", "Rarely"], p=p_att)
        att_freq_list.append(att_freq)
        
        # 2. Regularity
        if att_freq == "3-4 times a week":
            reg = np.random.choice(["Very regular", "Regular", "Sometimes regular"], p=[0.6, 0.3, 0.1])
        elif att_freq == "1-2 times a week":
            reg = np.random.choice(["Regular", "Sometimes regular", "Irregular"], p=[0.2, 0.6, 0.2])
        else:
            reg = np.random.choice(["Sometimes regular", "Irregular"], p=[0.25, 0.75])
        att_reg_list.append(reg)
        
        # 3. Classroom Participation
        p_part = [0.35 + 0.15*lat, 0.40, 0.20 - 0.1*lat, 0.05 - 0.05*lat]
        p_part = np.clip(p_part, 0.03, 0.95)
        p_part /= np.sum(p_part)
        part = np.random.choice(["Very actively", "Actively", "Sometimes", "Rarely"], p=p_part)
        part_list.append(part)
        
        # 4. Assignment Completion
        p_ass = [0.4 + 0.15*lat, 0.35, 0.2 - 0.1*lat, 0.05 - 0.05*lat]
        p_ass = np.clip(p_ass, 0.02, 0.95)
        p_ass /= np.sum(p_ass)
        assign = np.random.choice(["Always", "Often", "Sometimes", "Never"], p=p_ass)
        assign_list.append(assign)
        
        # 5. Miss Reason
        reasons = []
        if reg in ["Sometimes regular", "Irregular"]:
            candidates = ["Family responsibilities", "Health issues", "Transport problems", "Lack of study time", "Personal reasons"]
            if lat < -0.8:
                candidates.append("Lack of interest")
            chosen = list(np.random.choice(candidates, size=np.random.randint(1, 3), replace=False))
            reasons.extend(chosen)
        else:
            if np.random.rand() > 0.6:
                reasons.append(np.random.choice(["Health issues", "Personal reasons"]))
            else:
                reasons.append("None")
        reasons_miss_list.append("; ".join(set(reasons)))
        
        # 6. Topic Understanding Ease
        p_und = [0.2 + 0.1*lat, 0.4 + 0.05*lat, 0.3 - 0.05*lat, 0.08 - 0.05*lat, 0.02]
        p_und = np.clip(p_und, 0.02, 0.95)
        p_und /= np.sum(p_und)
        und = np.random.choice(["Very easy", "Easy", "Moderate", "Difficult", "Very Difficult"], p=p_und)
        topic_und_list.append(und)
        
        # 7. Teaching Rating & Help
        tr = np.random.choice(["Excellent", "Good", "Average", "Poor", "Very poor"], p=[0.45, 0.35, 0.15, 0.03, 0.02])
        teach_rating_list.append(tr)
        
        th = np.random.choice(["Always", "Often", "Sometimes", "Rarely", "Never"], p=[0.48, 0.32, 0.15, 0.03, 0.02])
        teach_help_list.append(th)
        
        # 8. Improvement & Confidence
        p_imp = np.array([0.35 + 0.1*lat, 0.38, 0.18, 0.05, 0.04])
        p_imp = np.clip(p_imp, 0.02, 0.95)
        p_imp = p_imp / np.sum(p_imp)
        imp = np.random.choice(["Improved significantly", "Improved", "Slightly improved", "No improvement", "Not sure"], p=p_imp)
        improvement_list.append(imp)
        
        p_conf = np.array([0.30 + 0.1*lat, 0.40, 0.20 - 0.05*lat, 0.10 - 0.05*lat])
        p_conf = np.clip(p_conf, 0.02, 0.95)
        p_conf = p_conf / np.sum(p_conf)
        conf = np.random.choice(["Very confident", "Confident", "Somewhat confident", "Not confident at all"], p=p_conf)
        confidence_list.append(conf)
        
        # 9. Digital Tech & Resources
        res_opts = ["Textbooks", "Notes/Study Material", "Computers", "Internet Resources", "Library", "Videos/Online Courses"]
        n_res = np.random.randint(2, 5)
        resources_used_list.append("; ".join(np.random.choice(res_opts, size=n_res, replace=False)))
        
        d_freq = np.random.choice(["Daily", "Several times a week", "Once a week", "Rarely", "Never"], p=[0.38, 0.35, 0.15, 0.08, 0.04])
        tech_freq_list.append(d_freq)
        
        dev = np.random.choice(["Smartphone", "Laptop", "Desktop", "Tablet"], p=[0.72, 0.16, 0.08, 0.04])
        primary_device_list.append(dev)
        
        res_avail_list.append(int(np.clip(round(3.8 + np.random.normal(0, 0.8)), 1, 5)))
        tech_comfort_list.append(int(np.clip(round(3.9 + np.random.normal(0, 0.9)), 1, 5)))
        
        dig_opts = ["Educational videos", "Online quizes", "E-books/PDF's", "Learning websites", "Mobile apps", "Online classes"]
        digital_content_list.append("; ".join(np.random.choice(dig_opts, size=np.random.randint(1, 4), replace=False)))
        
        # 10. Challenges & Difficulties
        ch_opts = ["Lack of study time", "Difficulty understanding topics", "Internet/technology problems", 
                   "Personal/family responsibilities", "Language difficulties", "Lack of learning materials", "None"]
        if lat < 0:
            ch_chosen = list(np.random.choice(ch_opts[:-1], size=np.random.randint(1, 3), replace=False))
        else:
            ch_chosen = [np.random.choice(ch_opts)]
        challenges_list.append("; ".join(ch_chosen))
        
        diff_f = np.random.choice(["Very often", "Often", "Sometimes", "Rarely", "Never"], 
                                  p=[0.06, 0.14, 0.48, 0.24, 0.08] if lat>=0 else [0.20, 0.35, 0.30, 0.12, 0.03])
        diff_freq_list.append(diff_f)
        
        ask_comfort_list.append(int(np.clip(round(3.7 + 0.4*lat + np.random.normal(0, 0.8)), 1, 5)))
        
        approach = np.random.choice(["Teacher", "Friends/classmates", "I usually study on my own", "Online Resources", "Family members"],
                                    p=[0.42, 0.28, 0.15, 0.10, 0.05])
        approach_list.append(approach)
        
        suggestions_list.append(np.random.choice(suggestion_pool))
        
    # Derived Numerical Risk Score & Target Classification
    # Higher risk points = more disengaged or struggling
    risk_points = np.zeros(n_samples)
    for idx in range(n_samples):
        pts = 0
        if att_freq_list[idx] == "Rarely": pts += 35
        elif att_freq_list[idx] == "1-2 times a week": pts += 15
        
        if att_reg_list[idx] == "Irregular": pts += 25
        elif att_reg_list[idx] == "Sometimes regular": pts += 12
        
        if part_list[idx] == "Rarely": pts += 20
        elif part_list[idx] == "Sometimes": pts += 8
        
        if assign_list[idx] in ["Never", "Sometimes"]: pts += 15
        
        if topic_und_list[idx] in ["Difficult", "Very Difficult"]: pts += 15
        if diff_freq_list[idx] in ["Very often", "Often"]: pts += 10
        if confidence_list[idx] == "Not confident at all": pts += 10
        if "Lack of interest" in reasons_miss_list[idx]: pts += 20
        if ask_comfort_list[idx] <= 2: pts += 10
        
        risk_points[idx] = pts
        
    risk_prob = np.clip(risk_points / 130.0 + np.random.normal(0, 0.04, size=n_samples), 0.03, 0.97)
    
    risk_levels = []
    for p in risk_prob:
        if p >= 0.52:
            risk_levels.append("High Risk")
        elif p >= 0.28:
            risk_levels.append("Moderate Risk")
        else:
            risk_levels.append("Low Risk")
            
    # Synthetic Quantitative Proxy Scores for deeper analytics
    estimated_attendance_pct = []
    estimated_quiz_score = []
    for idx in range(n_samples):
        if att_freq_list[idx] == "3-4 times a week":
            base_att = 85
        elif att_freq_list[idx] == "1-2 times a week":
            base_att = 60
        else:
            base_att = 35
        estimated_attendance_pct.append(int(np.clip(base_att + np.random.normal(0, 8), 10, 100)))
        
        if topic_und_list[idx] in ["Very easy", "Easy"]:
            base_score = 80
        elif topic_und_list[idx] == "Moderate":
            base_score = 65
        else:
            base_score = 42
        estimated_quiz_score.append(int(np.clip(base_score + np.random.normal(0, 10), 15, 100)))
        
    df = pd.DataFrame({
        "learner_id": learner_ids,
        "age_group": age_groups,
        "education_level": edu_levels,
        "gender": genders,
        "occupation_status": occupations,
        
        # Section 2: Attendance and participation
        "attendance_frequency": att_freq_list,
        "attendance_regularity": att_reg_list,
        "classroom_participation": part_list,
        "assignment_completion": assign_list,
        "reasons_miss_classes": reasons_miss_list,
        
        # Section 3: Learning experience and performance
        "topic_understanding_ease": topic_und_list,
        "teaching_method_rating": teach_rating_list,
        "teacher_help_frequency": teach_help_list,
        "performance_improvement": improvement_list,
        "confidence_level": confidence_list,
        
        # Section 4: Resources and Technology
        "resources_used": resources_used_list,
        "digital_device_frequency": tech_freq_list,
        "primary_device": primary_device_list,
        "resource_availability_rating": res_avail_list,
        "tech_comfort_rating": tech_comfort_list,
        "digital_content_type": digital_content_list,
        
        # Section 5: Challenges and Support
        "challenges_faced": challenges_list,
        "difficulty_frequency": diff_freq_list,
        "asking_teacher_comfort": ask_comfort_list,
        "support_approach": approach_list,
        "suggestions_feedback": suggestions_list,
        
        # Quantitative analytics proxies
        "attendance_pct_estimate": estimated_attendance_pct,
        "quiz_score_estimate": estimated_quiz_score,
        
        # Machine Learning Ground Truths
        "risk_probability": np.round(risk_prob, 3),
        "risk_level": risk_levels
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(out_dir, "survey_responses_pilot.csv")
    df = generate_survey_dataset(n_samples=320)
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} survey responses at {out_csv}")
    print(df["risk_level"].value_counts())

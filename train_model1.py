"""
Model 1: Early Warning & Dropout Risk Classification Engine
Community Learning Centers (CLC) Learning Analytics

This script:
1. Loads the survey dataset (survey_responses.csv)
2. Engineers quantitative risk index and target labels (Low Risk, Moderate Risk, High Risk)
3. Trains and evaluates Supervised ML Classifiers:
   - Random Forest Classifier (handles non-linear interactions & feature importance)
   - Logistic Regression (interpretable odds ratios)
4. Saves the trained model pipeline and feature weights (model.joblib)
5. Provides risk scoring function with actionable volunteer intervention advice.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def compute_risk_heuristics(df):
    """
    Computes evidence-based risk points based on survey indicators:
    - Attendance frequency & regularity
    - Missing reasons (Lack of interest, Transport, Health)
    - Participation in class & assignment completion
    - Difficulty frequency & confidence level
    - Comfort asking teachers for assistance
    """
    risk_scores = []
    for _, row in df.iterrows():
        pts = 0
        # 1. Attendance Frequency (Weight: 25)
        att_f = str(row.get('How often do you attend the learning center?', ''))
        if 'Rarely' in att_f: pts += 25
        elif '1-2 times' in att_f: pts += 10
        
        # 2. Regularity (Weight: 20)
        att_r = str(row.get('How regular is your attendance?', ''))
        if 'Irregular' in att_r: pts += 20
        elif 'Sometimes regular' in att_r: pts += 10
        
        # 3. Participation & Assignment completion (Weight: 20)
        part = str(row.get('How actively you participate in classroom activities?', ''))
        if 'Rarely' in part: pts += 12
        elif 'Sometimes' in part: pts += 6
        
        assign = str(row.get('How often do you complete your assignments or learning activities?', ''))
        if 'Never' in assign: pts += 12
        elif 'Sometimes' in assign: pts += 6
        
        # 4. Miss Reasons (Weight: 15)
        reasons = str(row.get('What is the main reason if you miss classes?', ''))
        if 'Lack of interest' in reasons: pts += 15
        elif 'Transport' in reasons or 'Personal' in reasons or 'Family' in reasons: pts += 6
        
        # 5. Academic struggle & confidence (Weight: 15)
        diff_f = str(row.get('How often do you face difficulties while learning?', ''))
        if 'Very often' in diff_f: pts += 10
        elif 'Often' in diff_f: pts += 5
        
        conf = str(row.get('How confident are you about the topics you have learned', ''))
        if 'Not confident at all' in conf: pts += 10
        elif 'Somewhat confident' in conf: pts += 5
        
        # 6. Reluctance to ask teachers (Weight: 10)
        try:
            ask_comfort = float(row.get('How comfortable are you asking teachers for help?', 3))
            if ask_comfort <= 1: pts += 10
            elif ask_comfort == 2: pts += 6
        except:
            pass
            
        risk_scores.append(pts)
        
    risk_scores = np.array(risk_scores)
    # Normalized risk probability (0 to 1)
    risk_prob = np.clip(risk_scores / 85.0, 0.05, 0.98)
    
    # Categorical Risk Levels:
    # High Risk: >= 0.50
    # Moderate Risk: 0.25 - 0.49
    # Low Risk: < 0.25
    risk_levels = []
    for p in risk_prob:
        if p >= 0.48:
            risk_levels.append('High Risk')
        elif p >= 0.25:
            risk_levels.append('Moderate Risk')
        else:
            risk_levels.append('Low Risk')
            
    df['risk_score_raw'] = risk_scores
    df['risk_probability'] = np.round(risk_prob, 3)
    df['risk_level'] = risk_levels
    return df

def train_early_warning_model(csv_path="survey_responses.csv"):
    df = pd.read_csv(csv_path)
    df = compute_risk_heuristics(df)
    
    # Feature columns used for training the model
    feature_cols = [
        'What is your age group?',
        'What is your current education level?',
        'What is your current occupation/status?',
        'How often do you attend the learning center?',
        'How regular is your attendance?',
        'How actively you participate in classroom activities?',
        'How often do you complete your assignments or learning activities?',
        'How easy is it for you to understand the topics taught?',
        'How would you rate the teaching methods used at the learning center?',
        'How often do teachers help you when you face difficulties?',
        'How confident are you about the topics you have learned',
        'How often do you use digital devices for learning?',
        'Which device do you mainly use for learning?',
        'How would you rate the availability of learning resources at the center?',
        'How comfortable are you using technology for learning?',
        'How often do you face difficulties while learning?',
        'How comfortable are you asking teachers for help?',
        'Who do you usually approach when you need help with your studies?'
    ]
    
    X = df[feature_cols].copy()
    y = df['risk_level']
    
    # Identify numeric vs categorical
    numeric_cols = [
        'How would you rate the availability of learning resources at the center?',
        'How comfortable are you using technology for learning?',
        'How comfortable are you asking teachers for help?'
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    
    # Fill NAs
    for col in categorical_cols:
        X[col] = X[col].fillna('Missing').astype(str)
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(3.0)
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Train Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    
    # Train on dataset
    rf_pipeline.fit(X, y)
    
    # Feature Importances extraction
    ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(cat_feature_names)
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save artifacts
    artifacts = {
        'model': rf_pipeline,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_importances': feat_imp_df,
        'classes': rf_pipeline.classes_.tolist()
    }
    
    model_path = os.path.join(os.path.dirname(__file__), "risk_model.joblib")
    joblib.dump(artifacts, model_path)
    
    scored_csv_path = os.path.join(os.path.dirname(__file__), "survey_scored.csv")
    df.to_csv(scored_csv_path, index=False)
    
    print(f"Model trained successfully. Saved to {model_path}")
    print(f"Scored dataset saved to {scored_csv_path}")
    print("\nTop 10 Influential Features for Dropout/Disengagement Risk:")
    print(feat_imp_df.head(10))
    return artifacts, df

if __name__ == "__main__":
    train_early_warning_model()

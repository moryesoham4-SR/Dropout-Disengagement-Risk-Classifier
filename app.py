import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="CLC Learning Analytics & Early Warning Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
    }
    .moderate-risk-badge {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
    }
    .low-risk-badge {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Load Scored Data and Trained Model
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "survey_scored.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "survey_responses.csv")
    df = pd.read_csv(data_path)
    
    # Anonymize identifier if email exists or generate Learner ID
    if 'Learner_ID' not in df.columns:
        df['Learner_ID'] = [f"CLC-L{101 + i}" for i in range(len(df))]
        
    return df

@st.cache_resource
def load_ml_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "risk_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model_artifacts = load_ml_model()

# Header
st.markdown('<div class="main-header">🎓 Community Learning Center: Learning Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Community Engagement Project</b> — Powered by Supervised Early Warning & Dropout Risk Classification Model</div>', unsafe_allow_html=True)

# Sidebar Filter & Navigation
st.sidebar.image("https://img.icons8.com/color/96/education.png", width=70)
st.sidebar.title("Navigation & Filters")
nav_selection = st.sidebar.radio(
    "Go to Module:",
    ["📊 Executive Overview & Analytics", 
     "🚨 Early Warning & At-Risk Roster (Model 1)", 
     "🧪 What-If Student Risk Simulator", 
     "💡 Learning Challenges & Feedback Insights",
     "📥 Data Exporter & Templates"]
)

# Global Filters in Sidebar
st.sidebar.subheader("Filter Survey Data")
selected_gender = st.sidebar.multiselect(
    "Gender:",
    options=df['What is your gender?'].dropna().unique(),
    default=df['What is your gender?'].dropna().unique()
)
selected_edu = st.sidebar.multiselect(
    "Education Level:",
    options=df['What is your current education level?'].dropna().unique(),
    default=df['What is your current education level?'].dropna().unique()
)
selected_occ = st.sidebar.multiselect(
    "Occupation Status:",
    options=df['What is your current occupation/status?'].dropna().unique(),
    default=df['What is your current occupation/status?'].dropna().unique()
)

filtered_df = df[
    (df['What is your gender?'].isin(selected_gender)) &
    (df['What is your current education level?'].isin(selected_edu)) &
    (df['What is your current occupation/status?'].isin(selected_occ))
]

# -------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & ANALYTICS
# -------------------------------------------------------------
if nav_selection == "📊 Executive Overview & Analytics":
    st.subheader("Key Educational Indicators & Impact Metrics")
    
    total_students = len(filtered_df)
    high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'High Risk'])
    mod_risk_count = len(filtered_df[filtered_df['risk_level'] == 'Moderate Risk'])
    low_risk_count = len(filtered_df[filtered_df['risk_level'] == 'Low Risk'])
    
    regular_attendance_pct = round((len(filtered_df[filtered_df['How regular is your attendance?'].isin(['Very regular', 'Regular'])]) / max(total_students, 1)) * 100, 1)
    improved_pct = round((len(filtered_df[filtered_df['Has your academic performance or skills improved after joining the learning center?'].isin(['Improved significantly', 'Improved'])]) / max(total_students, 1)) * 100, 1)
    
    # KPI Columns
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Surveyed Learners", f"{total_students}")
    with kpi2:
        st.metric("Regular Attendance Rate", f"{regular_attendance_pct}%")
    with kpi3:
        st.metric("Reported Improvement", f"{improved_pct}%")
    with kpi4:
        st.metric("🚨 High Risk (Flagged)", f"{high_risk_count}", delta=f"{round(high_risk_count/max(total_students,1)*100, 1)}%", delta_color="inverse")
    with kpi5:
        st.metric("⚠️ Moderate Risk", f"{mod_risk_count}", delta=f"{round(mod_risk_count/max(total_students,1)*100, 1)}%", delta_color="inverse")
        
    st.markdown("---")
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown("#### 1. Attendance Frequency vs Academic Improvement")
        fig1 = px.histogram(
            filtered_df, 
            x="How often do you attend the learning center?", 
            color="Has your academic performance or skills improved after joining the learning center?",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Prism,
            labels={"How often do you attend the learning center?": "Attendance Frequency"}
        )
        fig1.update_layout(xaxis_title="Attendance Frequency", yaxis_title="Number of Learners", legend_title="Skill Improvement")
        st.plotly_chart(fig1, use_container_width=True)
        
    with row1_c2:
        st.markdown("#### 2. Risk Level Distribution (Model 1 Classification)")
        risk_counts = filtered_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig2 = px.pie(
            risk_counts, 
            names='Risk Level', 
            values='Count',
            color='Risk Level',
            color_discrete_map={
                'High Risk': '#EF4444',
                'Moderate Risk': '#F59E0B',
                'Low Risk': '#10B981'
            },
            hole=0.45
        )
        st.plotly_chart(fig2, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown("#### 3. Primary Learning Device Used")
        device_counts = filtered_df['Which device do you mainly use for learning?'].value_counts().reset_index()
        device_counts.columns = ['Device', 'Learners']
        fig3 = px.bar(
            device_counts, 
            x='Device', 
            y='Learners', 
            color='Device',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    with row2_c2:
        st.markdown("#### 4. Comfort Asking Teachers for Help (1 to 5)")
        fig4 = px.histogram(
            filtered_df, 
            x="How comfortable are you asking teachers for help?", 
            color="risk_level",
            color_discrete_map={
                'High Risk': '#EF4444',
                'Moderate Risk': '#F59E0B',
                'Low Risk': '#10B981'
            },
            nbins=5
        )
        fig4.update_layout(xaxis_title="Comfort Rating (1=Low, 5=High)", yaxis_title="Learners")
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: EARLY WARNING & AT-RISK ROSTER (MODEL 1)
# -------------------------------------------------------------
elif nav_selection == "🚨 Early Warning & At-Risk Roster (Model 1)":
    st.subheader("Early Warning System: Learner Disengagement & Dropout Tracker")
    st.markdown("""
    This table uses **Model 1 (Random Forest Classifier & Rule-based Calibration)** to assign a real-time 
    **Risk Level** and **Early Warning Probability** to every learner. Coordinators and volunteers can filter 
    students to prioritize interventions.
    """)
    
    # Filter by Risk Level
    risk_filter = st.radio(
        "Filter Roster by Risk Level:",
        ["All Learners", "🚨 High Risk Only", "⚠️ Moderate Risk Only", "✅ Low Risk Only"],
        horizontal=True
    )
    
    display_df = filtered_df.copy()
    if risk_filter == "🚨 High Risk Only":
        display_df = display_df[display_df['risk_level'] == 'High Risk']
    elif risk_filter == "⚠️ Moderate Risk Only":
        display_df = display_df[display_df['risk_level'] == 'Moderate Risk']
    elif risk_filter == "✅ Low Risk Only":
        display_df = display_df[display_df['risk_level'] == 'Low Risk']
        
    # Table Summary
    st.write(f"Showing **{len(display_df)}** learners matching current criteria:")
    
    table_cols = [
        'Learner_ID',
        'risk_level',
        'risk_probability',
        'How often do you attend the learning center?',
        'How regular is your attendance?',
        'How actively you participate in classroom activities?',
        'What is the main reason if you miss classes?',
        'How comfortable are you asking teachers for help?',
        'Who do you usually approach when you need help with your studies?'
    ]
    
    # Clean column display names
    renamed_cols = {
        'risk_level': 'Predicted Risk',
        'risk_probability': 'Dropout Prob.',
        'How often do you attend the learning center?': 'Attendance Freq.',
        'How regular is your attendance?': 'Regularity',
        'How actively you participate in classroom activities?': 'Class Participation',
        'What is the main reason if you miss classes?': 'Absence Reason',
        'How comfortable are you asking teachers for help?': 'Teacher Comfort (1-5)',
        'Who do you usually approach when you need help with your studies?': 'Support Source'
    }
    
    st.dataframe(
        display_df[table_cols].rename(columns=renamed_cols),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Model 1: Feature Importance Analysis")
    st.markdown("Which survey indicators most strongly predict learner dropout/disengagement risk?")
    
    if model_artifacts and 'feature_importances' in model_artifacts:
        feat_df = model_artifacts['feature_importances'].head(10).copy()
        fig_feat = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            title="Top 10 Drivers of Disengagement Risk (Random Forest Weights)"
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Feature importance data is loading.")

# -------------------------------------------------------------
# TAB 3: WHAT-IF STUDENT RISK SIMULATOR
# -------------------------------------------------------------
elif nav_selection == "🧪 What-If Student Risk Simulator":
    st.subheader("Predict Learner Risk for New Survey Submission / Student Intake")
    st.markdown("Enter student responses below to calculate real-time **Disengagement & Dropout Probability** and view actionable volunteer advice.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("##### 👤 Learner Background & Attendance")
        sim_age = st.selectbox("Age Group:", ["Below 15", "15-18", "19-22", "Above 22"], index=2)
        sim_edu = st.selectbox("Current Education Level:", ["Primary", "Secondary", "Higher Secondary", "Undergraduate", "Other"], index=3)
        sim_occ = st.selectbox("Current Occupation:", ["Student", "Working", "Self-employed", "Not currently working"], index=0)
        sim_att_freq = st.selectbox("Center Attendance Frequency:", ["3-4 times a week", "1-2 times a week", "Rarely"], index=0)
        sim_att_reg = st.selectbox("Attendance Regularity:", ["Very regular", "Regular", "Sometimes regular", "Irregular"], index=1)
        sim_miss_reason = st.selectbox("Main Reason for Missing Classes:", ["None", "Personal reasons", "Health issues", "Family responsibilities", "Transport problems", "Lack of interest"], index=2)
        sim_part = st.selectbox("Classroom Participation:", ["Very actively", "Actively", "Sometimes", "Rarely"], index=1)
        sim_assign = st.selectbox("Assignment Completion:", ["Always", "Often", "Sometimes", "Never"], index=0)

    with col_b:
        st.markdown("##### 📚 Academics, Resources & Mentorship")
        sim_und = st.selectbox("Topic Understanding Ease:", ["Very easy", "Easy", "Moderate", "Difficult", "Very Difficult"], index=1)
        sim_teach_rate = st.selectbox("Teaching Methods Rating:", ["Excellent", "Good", "Average", "Poor", "Very poor"], index=1)
        sim_teach_help = st.selectbox("How often teachers help when facing difficulties:", ["Always", "Often", "Sometimes", "Rarely", "Never"], index=0)
        sim_conf = st.selectbox("Confidence in Learned Topics:", ["Very confident", "Confident", "Somewhat confident", "Not confident at all"], index=1)
        sim_dev = st.selectbox("Primary Device Used:", ["Smartphone", "Laptop", "Desktop", "Tablet"], index=0)
        sim_res_rate = st.slider("Resource Availability Rating (1 to 5):", 1, 5, 4)
        sim_tech_rate = st.slider("Comfort Using Technology (1 to 5):", 1, 5, 4)
        sim_diff_freq = st.selectbox("How often do you face difficulties:", ["Very often", "Often", "Sometimes", "Rarely", "Never"], index=2)
        sim_ask_comfort = st.slider("Comfort Asking Teachers for Help (1 to 5):", 1, 5, 4)
        sim_approach = st.selectbox("Who do you usually approach for help:", ["Teacher", "Friends/classmates", "Family members", "Online Resources", "I usually study on my own"], index=0)

    st.markdown("---")
    if st.button("🔮 Run Risk Prediction Model", type="primary", use_container_width=True):
        # Heuristic Risk Calculation matching Model 1
        pts = 0
        if "Rarely" in sim_att_freq: pts += 25
        elif "1-2 times" in sim_att_freq: pts += 10
        
        if "Irregular" in sim_att_reg: pts += 20
        elif "Sometimes regular" in sim_att_reg: pts += 10
        
        if "Rarely" in sim_part: pts += 12
        elif "Sometimes" in sim_part: pts += 6
        
        if "Never" in sim_assign: pts += 12
        elif "Sometimes" in sim_assign: pts += 6
        
        if "Lack of interest" in sim_miss_reason: pts += 15
        elif sim_miss_reason in ["Transport problems", "Personal reasons", "Family responsibilities"]: pts += 6
        
        if "Very often" in sim_diff_freq: pts += 10
        elif "Often" in sim_diff_freq: pts += 5
        
        if "Not confident at all" in sim_conf: pts += 10
        elif "Somewhat confident" in sim_conf: pts += 5
        
        if sim_ask_comfort <= 1: pts += 10
        elif sim_ask_comfort == 2: pts += 6
        
        prob = np.clip(pts / 85.0, 0.05, 0.98)
        
        if prob >= 0.48:
            pred_level = "High Risk"
            alert_type = st.error
            badge_color = "#EF4444"
        elif prob >= 0.25:
            pred_level = "Moderate Risk"
            alert_type = st.warning
            badge_color = "#F59E0B"
        else:
            pred_level = "Low Risk"
            alert_type = st.success
            badge_color = "#10B981"
            
        r_c1, r_c2 = st.columns([1, 2])
        with r_c1:
            st.metric("Predicted Status", pred_level)
            st.metric("Estimated Dropout Probability", f"{int(prob * 100)}%")
            
        with r_c2:
            st.markdown(f"### Assessment: <span style='color:{badge_color}'>{pred_level}</span>", unsafe_allow_html=True)
            if pred_level == "High Risk":
                st.markdown("""
                > **Urgent Coordinator Actions Required:**
                > 1. **Immediate Volunteer Outreach**: Schedule a 1-on-1 informal check-in within 48 hours.
                > 2. **Address Root Cause**: Learner is facing friction with attendance or low teacher comfort. Assign a peer study buddy.
                > 3. **Flexible Study Support**: Provide offline summary notes if digital access or attendance is constrained.
                """)
            elif pred_level == "Moderate Risk":
                st.markdown("""
                > **Proactive Support Recommendations:**
                > 1. **Periodic Check-In**: Check assignment completion at the end of the week.
                > 2. **Encourage Doubts**: Dedicate 10 minutes before class for low-pressure doubt-clearing sessions.
                """)
            else:
                st.markdown("""
                > **Stable Engagement Status:**
                > - Learner exhibits strong attendance consistency and healthy classroom participation.
                > - Opportunity: Invite learner to be a **peer mentor** for struggling classmates.
                """)

# -------------------------------------------------------------
# TAB 4: CHALLENGES & FEEDBACK INSIGHTS
# -------------------------------------------------------------
elif nav_selection == "💡 Learning Challenges & Feedback Insights":
    st.subheader("Field Challenges & Grassroots Learner Feedback")
    st.markdown("Direct qualitative and quantitative feedback gathered from community learners.")
    
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("#### Most Common Learning Challenges")
        # Split multi-select challenges
        all_challenges = []
        for ch in filtered_df['What challenges do you commonly face while learning?'].dropna():
            items = [item.strip() for item in ch.split(',')]
            all_challenges.extend(items)
            
        ch_series = pd.Series(all_challenges).value_counts().head(7).reset_index()
        ch_series.columns = ['Challenge', 'Count']
        fig_ch = px.bar(ch_series, x='Count', y='Challenge', orientation='h', color='Count', color_continuous_scale='Reds')
        fig_ch.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_ch, use_container_width=True)
        
    with col_ch2:
        st.markdown("#### Primary Support Contact When Needing Help")
        app_counts = filtered_df['Who do you usually approach when you need help with your studies?'].value_counts().reset_index()
        app_counts.columns = ['Support Source', 'Count']
        fig_app = px.pie(app_counts, names='Support Source', values='Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_app, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Direct Learner Suggestions & Voices from the Field")
    suggestions_df = filtered_df[['What is your age group?', 'What is your current occupation/status?', 'Please share any other difficulty or suggestion related to your learning.']].dropna()
    suggestions_df = suggestions_df[~suggestions_df['Please share any other difficulty or suggestion related to your learning.'].isin(['None', 'Nothing', 'NA', 'No', '.', '..', '...'])]
    
    for _, row in suggestions_df.head(10).iterrows():
        st.info(f"🗣️ *\"{row['Please share any other difficulty or suggestion related to your learning.']}\"*  \n— Learner ({row['What is your age group?']}, {row['What is your current occupation/status?']})")

# -------------------------------------------------------------
# TAB 5: DATA EXPORTER & TEMPLATES
# -------------------------------------------------------------
elif nav_selection == "📥 Data Exporter & Templates":
    st.subheader("Data Management & Handover")
    st.markdown("""
    In accordance with the **Community Engagement Project Objectives (Phase 5 Handover)**, 
    center coordinators can export the processed dataset with risk classifications or download standard entry templates.
    """)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Export Current Scored Dataset")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Learner Records (CSV)",
            data=csv_data,
            file_name="clc_learner_analytics_scored.csv",
            mime="text/csv",
            type="primary"
        )
        st.caption("Includes demographic fields, survey responses, computed risk points, and Model 1 risk classifications.")
        
    with col_d2:
        st.markdown("#### Download Standard Data Collection Template")
        blank_cols = [c for c in df.columns if not c.startswith('risk_') and c != 'Learner_ID']
        blank_template = pd.DataFrame(columns=blank_cols)
        blank_csv = blank_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Blank Survey Collection Template (CSV)",
            data=blank_csv,
            file_name="clc_survey_blank_template.csv",
            mime="text/csv"
        )
        st.caption("Standardized blank CSV template for volunteers to digitize subsequent batches of community learners.")

st.markdown("---")
st.caption("Community Engagement Project | Learning Analytics Dashboard for Non-Formal Education Centers | MBA Analytics & Management Information Systems")

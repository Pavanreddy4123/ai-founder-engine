import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Founder Engine", layout="centered")

st.title("🚀 AI Founder Compatibility & Equity Intelligence Engine")
st.markdown("Analyze co-founder compatibility using AI + ML")

# INPUT
st.header("👤 Founder 1")
f1_exp = st.slider("Experience (Years)", 0, 15, 3)
f1_time = st.slider("Time Commitment (%)", 0, 100, 60)
f1_risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])
f1_skill = st.selectbox("Skill", ["Tech", "Business", "Marketing"])

st.header("👤 Founder 2")
f2_exp = st.slider("Experience (Years)", 0, 15, 2)
f2_time = st.slider("Time Commitment (%)", 0, 100, 50)
f2_risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])
f2_skill = st.selectbox("Skill", ["Tech", "Business", "Marketing"])

if st.button("🔍 Run AI Analysis"):

    # Feature Engineering
    risk_map = {"Low": 1, "Medium": 2, "High": 3}
    skill_map = {"Tech": 1, "Business": 2, "Marketing": 3}

    risk_diff = abs(risk_map[f1_risk] - risk_map[f2_risk])
    skill_diff = abs(skill_map[f1_skill] - skill_map[f2_skill])
    exp_diff = abs(f1_exp - f2_exp)
    time_diff = abs(f1_time - f2_time)

    # Compatibility Score
    compatibility = max(0, 100 - (risk_diff*20 + skill_diff*10 + exp_diff*3 + time_diff*0.5))

    # ML Model
    X = np.array([[10,10,2,10],[20,20,1,5],[5,30,2,20],[15,10,3,40],[30,5,1,10],[10,25,2,15]])
    y = np.array([1,1,0,0,1,0])

    model = LogisticRegression()
    model.fit(X, y)

    pred = model.predict([[risk_diff*10, skill_diff*10, exp_diff, time_diff]])[0]
    prob = model.predict_proba([[risk_diff*10, skill_diff*10, exp_diff, time_diff]])[0][1]

    # Equity
    f1_total = f1_exp + f1_time
    f2_total = f2_exp + f2_time
    total = f1_total + f2_total if (f1_total + f2_total) != 0 else 1

    f1_eq = (f1_total / total) * 100
    f2_eq = (f2_total / total) * 100

    # Risk
    if risk_diff > 1:
        risk_level = "High"
    elif time_diff > 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # OUTPUT
    st.header("📊 Results")
    st.metric("Compatibility Score", f"{round(compatibility,2)} / 100")
    st.progress(int(compatibility))

    st.subheader("💰 Equity Split")
    st.write(f"Founder 1: {round(f1_eq,1)}%")
    st.write(f"Founder 2: {round(f2_eq,1)}%")

    st.subheader("🤖 ML Prediction")
    st.write(f"Success Probability: {round(prob*100,2)}%")

    st.subheader("⚠️ Risk Level")
    st.write(risk_level)

    # Graph 1
    st.subheader("📈 Compatibility vs Success")
    x = np.linspace(0,100,10)
    y = x * 0.9
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("Compatibility")
    plt.ylabel("Success")
    st.pyplot(plt)

    # Graph 2
    st.subheader("📉 Equity vs Conflict")
    x2 = np.linspace(0,100,10)
    y2 = 100 - x2
    plt.figure()
    plt.plot(x2,y2)
    plt.xlabel("Equity Fairness")
    plt.ylabel("Conflict")
    st.pyplot(plt)
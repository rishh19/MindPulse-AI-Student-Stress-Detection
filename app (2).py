import streamlit as st
import numpy as np
import joblib
import re
import time
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MindPulse Student AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE
# -----------------------------------------------------------------------------
default_values = {
    'study': 4, 'attendance': 75, 'sleep': 6, 'screen': 5,
    'pressure': 3, 'anxiety': 4, 'exercise': 1, 'social': 2,
    'journal_text': ""
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

def set_demo_data(level):
    if level == "low":
        st.session_state.study = 3
        st.session_state.attendance = 90
        st.session_state.sleep = 8
        st.session_state.screen = 2
        st.session_state.pressure = 2
        st.session_state.anxiety = 2
        st.session_state.exercise = 2
        st.session_state.social = 3
    elif level == "moderate":
        st.session_state.study = 5
        st.session_state.attendance = 75
        st.session_state.sleep = 6
        st.session_state.screen = 5
        st.session_state.pressure = 3
        st.session_state.anxiety = 5
        st.session_state.exercise = 1
        st.session_state.social = 2
    elif level == "high":
        st.session_state.study = 9
        st.session_state.attendance = 40
        st.session_state.sleep = 3
        st.session_state.screen = 8
        st.session_state.pressure = 5
        st.session_state.anxiety = 9
        st.session_state.exercise = 0
        st.session_state.social = 0

def clear_journal():
    st.session_state.journal_text = ""

def reset_lifestyle():
    for key, val in default_values.items():
        if key != 'journal_text':
            st.session_state[key] = val

# -----------------------------------------------------------------------------
# 3. CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        color: #333;
    }
    
    .stApp {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        background-attachment: fixed;
    }

    /* Cards */
    div.stTabs, div.css-1r6slb0, form {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: none;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }

    h1, h2, h3 { color: #2d3436 !important; font-weight: 800; }
    
    /* Input Fields */
    .stNumberInput input, .stTextInput input, .stTextArea textarea {
        background-color: #f1f2f6 !important;
        color: #2d3436 !important;
        border-radius: 12px;
        border: 2px solid #dfe6e9;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        height: 50px; /* Fixed height for consistency */
    }

    /* Result Card */
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-bottom: 8px solid #ddd;
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer { text-align: center; margin-top: 50px; color: #555; }
    #MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. LOAD MODELS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("numeric_stress_model.pkl"):
            models['numeric'] = joblib.load("numeric_stress_model.pkl")
        
        if os.path.exists("scaler (1).pkl"):
            models['scaler'] = joblib.load("scaler (1).pkl")
        elif os.path.exists("scaler.pkl"):
            models['scaler'] = joblib.load("scaler.pkl")
            
        if os.path.exists("nlp_stress_model.pkl"):
            models['nlp'] = joblib.load("nlp_stress_model.pkl")
        if os.path.exists("nlp_vectorizer.pkl"):
            models['vectorizer'] = joblib.load("nlp_vectorizer.pkl")
        return models
    except Exception as e:
        return None

models = load_models()

# -----------------------------------------------------------------------------
# 5. LOGIC HELPERS
# -----------------------------------------------------------------------------
def clean_text(t):
    return re.sub('[^a-zA-Z]', ' ', str(t)).lower()

def get_student_result(label):
    lbl = str(label).lower()
    
    # 3-Class Logic: High, Low, Moderate
    
    if "high" in lbl:
        return {
            "color": "#ff6b6b", "bg": "#ffeaea", "icon": "🌋",
            "title": "High Stress Detected",
            "msg": "Your markers indicate burnout risk. Please prioritize rest.",
            "actions": [
                "🛑 **STOP WORK:** Take a 20-min brain break.",
                "🌬️ **BREATHE:** 4-7-8 method to lower anxiety.",
                "🗣️ **SUPPORT:** Call a friend or counselor.",
                "💧 **HYDRATE:** Drink cold water immediately."
            ],
            "anim": "snow"
        }
    elif "moderate" in lbl:
        return {
            "color": "#feca57", "bg": "#fff9e6", "icon": "🚧",
            "title": "Moderate Pressure",
            "msg": "You are handling the semester, but don't let it pile up.",
            "actions": [
                "🍅 **POMODORO:** Study 25 min, Break 5 min.",
                "🎵 **LO-FI:** Use instrumental music to focus.",
                "📝 **PLAN:** Write down top 3 tasks only.",
                "🚶 **WALK:** A 5-min walk resets focus."
            ],
            "anim": "toast"
        }
    else: # Low/Optimal
        return {
            "color": "#1dd1a1", "bg": "#e3fdf5", "icon": "🎓",
            "title": "Optimal / Low Stress",
            "msg": "You are in the 'Flow State'. Great balance!",
            "actions": [
                "🚀 **GOAL:** Tackle your hardest subject now.",
                "✨ **SOCIAL:** Study groups can help.",
                "🥗 **FUEL:** Eat brain food (nuts/fruits).",
                "🛌 **SLEEP:** Maintain this healthy cycle."
            ],
            "anim": "balloons"
        }

# -----------------------------------------------------------------------------
# 6. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3403/3403567.png", width=80)
    st.title("MindPulse AI")
    st.write("Academic Stress Prediction System")
    
    st.markdown("---")
    st.subheader("🧪 Developer Mode")
    st.caption("Auto-fill data to test specific outcomes:")
    
    # Uniform Buttons
    if st.button("🟢 Test Low Stress", use_container_width=True):
        set_demo_data("low")
        st.rerun()
    if st.button("🟠 Test Moderate Stress", use_container_width=True):
        set_demo_data("moderate")
        st.rerun()
    if st.button("🔴 Test High Stress", use_container_width=True):
        set_demo_data("high")
        st.rerun()

# -----------------------------------------------------------------------------
# 7. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🧠 MindPulse Student AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; margin-bottom: 30px;'>Academic Stress Prediction & Management System</p>", unsafe_allow_html=True)

if models is None or len(models) < 4:
    st.error("⚠️ **System Error:** Model files missing. Please ensure `.pkl` files are in the folder.")
else:
    tab1, tab2 = st.tabs(["📊 Lifestyle Check", "💬 AI Journal"])

    # --- TAB 1: LIFESTYLE ---
    with tab1:
        st.write("### ☀️ Student Daily Habits")
        
        with st.form("lifestyle_form"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**📚 Academics**")
                st.number_input("Study (Hrs)", 0, 24, key='study')
                st.number_input("Attendance %", 0, 100, key='attendance')
            with c2:
                st.markdown("**💤 Physiology**")
                st.number_input("Sleep (Hrs)", 0, 24, key='sleep')
                st.number_input("Screen Time", 0, 24, key='screen')
            with c3:
                st.markdown("**🧠 Psychology**")
                st.number_input("Exam Pressure (1-5)", 1, 5, key='pressure')
                st.number_input("Anxiety (1-10)", 1, 10, key='anxiety')
            with c4:
                st.markdown("**🏃 Activity**")
                st.number_input("Exercise (Hrs)", 0, 5, key='exercise')
                st.number_input("Social (Hrs)", 0, 10, key='social')

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Button Layout: Analyze (Big), Reset (Small)
            b_col1, b_col2 = st.columns([3, 1])
            with b_col1:
                submit = st.form_submit_button("✨ Analyze Lifestyle Stress", use_container_width=True, type="primary")
            with b_col2:
                reset_btn = st.form_submit_button("🔄 Reset", use_container_width=True, on_click=reset_lifestyle)

        if submit:
            with st.spinner("Analyzing academic markers..."):
                time.sleep(0.5)
                features = np.array([[
                    st.session_state.study, st.session_state.sleep, st.session_state.screen, 
                    st.session_state.exercise, st.session_state.attendance, st.session_state.pressure, 
                    st.session_state.social, st.session_state.anxiety
                ]])
                
                try:
                    scaler = models['scaler']
                    if scaler.n_features_in_ == 8:
                        scaled_f = scaler.transform(features)
                        pred = models['numeric'].predict(scaled_f)[0]
                        
                        # 3-CLASS MAPPING (Based on your feedback)
                        labels = ["High Stress", "Low Stress", "Moderate Stress"]
                        
                        try: res_label = labels[int(pred)]
                        except: res_label = str(pred)
                            
                        res = get_student_result(res_label)
                        
                        st.markdown(f"""
                        <div class="result-card" style="border-bottom: 8px solid {res['color']}; background: {res['bg']};">
                            <h1 style="color: {res['color']}; font-size: 60px; margin:0;">{res['icon']}</h1>
                            <h2 style="color: {res['color']}; margin: 10px 0;">{res['title']}</h2>
                            <p style="font-size: 18px; color: #555;">{res['msg']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 💡 Student Suggestions")
                        cols = st.columns(2)
                        for i, act in enumerate(res['actions']):
                            with cols[i % 2]: st.info(act)

                        if res['anim'] == "snow": st.snow()
                        elif res['anim'] == "balloons": st.balloons()
                        elif res['anim'] == "toast": st.toast("Moderate Stress Detected", icon="🧘")
                        
                    else:
                        st.error(f"Model mismatch! Expecting {scaler.n_features_in_} inputs.")
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

    # --- TAB 2: JOURNAL ---
    with tab2:
        st.write("### 📝 Emotional Check-in")
        
        st.text_area("Dear Diary...", height=150, key="journal_text", placeholder="I am feeling overwhelmed...")
        
        # Consistent Button Sizing
        jb1, jb2 = st.columns([3, 1])
        with jb1: analyze_btn = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
        with jb2: st.button("🗑️ Clear", on_click=clear_journal, use_container_width=True)

        if analyze_btn and st.session_state.journal_text:
            text_in = st.session_state.journal_text.lower()
            
            # 1. TRIGGER WORDS CHECK (CRISIS)
            crisis_triggers = ["die", "suicide", "kill", "death", "end it", "hurt myself"]
            
            # 2. POSITIVE WORDS CHECK (FORCE LOW STRESS)
            positive_triggers = ["nice", "happily", "happy", "good", "great", "awesome", "calm", "relax", "joy", "excellent", "fine"]

            if any(x in text_in for x in crisis_triggers):
                st.error("🚨 **CRITICAL ALERT: IMMEDIATE SUPPORT NEEDED**")
                st.markdown("""
                <div style="background-color: #ffebee; padding: 25px; border-radius: 15px; border-left: 8px solid #c62828;">
                    <h3 style="color: #c62828; margin-top: 0;">You are not alone.</h3>
                    <p>We hear that you are in pain. There are people who want to listen.</p>
                    <ul>
                        <li><b>Vandrevala Foundation:</b> 1860-266-2345</li>
                        <li><b>iCall Helpline:</b> 9152987821</li>
                        <li><b>Kiran (Govt):</b> 1800-599-0019</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. IF POSITIVE WORDS DETECTED -> FORCE LOW STRESS
            elif any(x in text_in for x in positive_triggers):
                res = get_student_result("Low Stress")
                
                st.markdown(f"""
                <div class="result-card" style="border-bottom: 8px solid {res['color']}; background: {res['bg']};">
                    <h2 style="color: {res['color']}; margin: 10px 0;">Sentiment: {res['title']}</h2>
                    <p><i>(Positive keyword detected)</i></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Coping Strategy")
                cols = st.columns(2)
                for i, act in enumerate(res['actions']):
                    with cols[i % 2]: st.info(act)
                st.balloons()
            
            # 4. ELSE -> RUN NLP MODEL
            else:
                try:
                    clean = clean_text(text_in)
                    vect_t = models['vectorizer'].transform([clean])
                    pred = models['nlp'].predict(vect_t)[0]
                    
                    # Ensure prediction maps to your 3 classes
                    # Assuming NLP model output aligns with numeric, or map explicitly if needed
                    # For now, we pass the raw prediction to get_student_result which handles strings/ints
                    
                    res = get_student_result(pred)
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-bottom: 8px solid {res['color']}; background: {res['bg']};">
                        <h2 style="color: {res['color']}; margin: 10px 0;">Sentiment: {res['title']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 💡 Coping Strategy")
                    cols = st.columns(2)
                    for i, act in enumerate(res['actions']):
                        with cols[i % 2]: st.info(act)
                        
                    if res['anim'] == "snow": st.snow()
                    elif res['anim'] == "balloons": st.balloons()
                    elif res['anim'] == "toast": st.toast(f"Sentiment: {res['title']}", icon="📝")
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ by Rishav Kumar Shrivastava | Aspiring Data Scientist</div>", unsafe_allow_html=True)
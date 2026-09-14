# 🧠 MindPulse AI — Student Stress Detection & Wellness Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-success)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Naive%20Bayes-yellow)

MindPulse AI is a machine learning and NLP-based Streamlit application designed to help students assess stress levels using **academic and lifestyle information** as well as **free-text journal entries**.

The project combines two machine learning pipelines:

- **Numerical stress prediction** using Logistic Regression
- **Text-based stress detection** using TF-IDF and Multinomial Naive Bayes

The application also provides general wellness suggestions based on the predicted stress level.

> **Note:** This project is intended for educational and stress-awareness purposes. It is not a medical or clinical diagnostic system.

---

## 🚀 Live Demo

**Streamlit Application:**  
https://mindpulse-ai-student-stress-detection.streamlit.app/

---

## 🌟 Project Overview

Students can experience stress due to academic workload, examination pressure, sleep patterns, screen time, anxiety, and other lifestyle factors.

MindPulse AI explores how machine learning can be used to analyze these factors and provide a simple stress-level assessment.

The application provides two main ways to interact with the system:

### 📊 1. Lifestyle Check

Users enter information related to their academic and lifestyle habits.

The numerical machine learning pipeline processes these inputs and predicts the student's stress level.

### 📝 2. AI Journal

Users can enter a short free-text description of how they are feeling.

The NLP pipeline processes the text and predicts an associated stress level.

---

# 🧠 AI Architecture

MindPulse AI uses two separate machine learning pipelines.

## 1. Numerical Stress Prediction

The numerical prediction pipeline follows:

```text
User Input
    ↓
Feature Scaling
    ↓
Logistic Regression
    ↓
Stress Level Prediction
    ↓
Wellness Suggestions
```

### Input Features

| Feature | Description |
|---|---|
| Study Hours | Daily study time |
| Sleep Hours | Average sleep duration |
| Screen Time | Daily screen/device usage |
| Attendance | Academic attendance |
| Exam Pressure | Exam-related pressure level |
| Anxiety Level | Self-reported anxiety level |
| Exercise Hours | Daily physical activity |
| Social Interaction | Level of social interaction |

### Model Selection

During experimentation, multiple classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Logistic Regression was selected as the final numerical prediction model based on the evaluation results obtained during experimentation.

### Evaluation Results

The selected Logistic Regression model achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 98.75% |
| Precision | 98.77% |
| Recall | 98.75% |
| F1-Score | 98.66% |

> These metrics are based on the project's experimental evaluation and should not be interpreted as clinical or real-world diagnostic accuracy.

---

# 📝 2. NLP-Based Stress Detection

The second component analyzes free-text journal entries.

### NLP Pipeline

```text
Journal Entry
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Multinomial Naive Bayes
      ↓
Stress Level Prediction
```

### Techniques Used

- Text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes classification

The NLP model predicts stress-level categories such as:

- Low
- Medium
- High

---

# 🖥️ Application Modules

## 📊 Lifestyle Check

Allows students to enter academic, psychological, physiological, and activity-related information and receive a predicted stress level.

## 📝 AI Journal

Allows users to describe their thoughts or feelings using free text.

The NLP model analyzes the journal entry and predicts a corresponding stress level.

## 🌱 Wellness Suggestions

Based on the predicted stress level, the application provides general suggestions related to:

- Study techniques
- Breaks and physical activity
- Task planning
- Focus and relaxation

## 🧪 Developer Demo Mode

The application includes developer testing controls that allow different stress-level scenarios to be demonstrated quickly:

- Low Stress
- Moderate Stress
- High Stress

This makes it easier to test and demonstrate the application's different output states.

## 🆘 Support Information

The application also provides support and helpline information for situations where additional assistance may be appropriate.

---

# 🏗️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Numerical Model | Logistic Regression |
| NLP Model | Multinomial Naive Bayes |
| Text Features | TF-IDF |
| Data Processing | Pandas, NumPy |
| Web Application | Streamlit |
| Model Serialization | Pickle |
| Development | Jupyter Notebook |

---

# 📂 Project Structure

```text
MindPulse-AI/
│
├── app.py
│
├── numeric_stress_model.pkl
├── scaler.pkl
│
├── nlp_stress_model.pkl
├── nlp_vectorizer.pkl
│
├── stressData.csv
├── stress_text_data.csv
│
├── requirements.txt
└── README.md
```

### Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `numeric_stress_model.pkl` | Trained numerical stress prediction model |
| `scaler.pkl` | Feature scaler used for numerical inputs |
| `nlp_stress_model.pkl` | Trained NLP classification model |
| `nlp_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `stressData.csv` | Numerical stress dataset |
| `stress_text_data.csv` | Text-based stress dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/rishh19/MindPulse-AI-Student-Stress-Detection.git
cd MindPulse-AI-Student-Stress-Detection
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔄 End-to-End Workflow

## Numerical Stress Prediction

```text
Student enters lifestyle & academic information
                    ↓
              Feature Scaling
                    ↓
          Logistic Regression
                    ↓
          Stress Level Prediction
                    ↓
           Wellness Suggestions
```

## Journal Stress Detection

```text
Student enters journal text
              ↓
       Text Preprocessing
              ↓
        TF-IDF Vectorizer
              ↓
   Multinomial Naive Bayes
              ↓
      Stress Prediction
```

---

# 🎯 Use Cases

MindPulse AI can serve as an educational demonstration of:

- Machine learning classification
- NLP-based text classification
- Feature preprocessing
- Model comparison
- TF-IDF text representation
- Integrating multiple ML pipelines into one application
- Streamlit application development
- Student stress-awareness applications

---

# 🔮 Future Improvements

Possible improvements include:

- Training and validating models on larger and more diverse datasets
- Improving model generalization and error analysis
- Adding prediction explainability
- Improving personalization of wellness suggestions
- Adding historical stress tracking
- Adding student analytics dashboards
- Exploring transformer-based NLP models
- Adding secure user authentication and data handling

---

# 👨‍💻 Author

**Rishav Kumar Shrivastava**

Computer Science & Engineering Student  
Machine Learning & Data Science Enthusiast

GitHub:  
https://github.com/rishh19

---

# ⚠️ Disclaimer

MindPulse AI is an **educational machine learning project** intended for stress-awareness and demonstration purposes.

It is **not a medical, psychological, or clinical diagnostic tool** and should not be used as a substitute for professional medical or mental-health advice.

If someone is experiencing serious mental-health difficulties or an emergency, they should seek assistance from a qualified healthcare professional or appropriate emergency/support service.

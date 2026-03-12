from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

diabetes_model = pickle.load(open("diabetes.pkl","rb"))
heart_model = pickle.load(open("heart.pkl","rb"))
kidney_model = pickle.load(open("kidney.pkl","rb"))
liver_model = pickle.load(open("liver.pkl","rb"))
lung_model = pickle.load(open("lung_cancer.pkl","rb"))


# HOME =================

@app.route("/")
def home():
    return render_template("home.html")



@app.route("/contact_doctor", methods=["GET", "POST"])
def contact_doctor():
    recommendation = ""
    notes = ""
    doctors = []

    if request.method == "POST":
        description = (request.form.get("description") or "").lower()

        # Simple rules to map description to specialist and example doctors.
        def set_doctors(specialty_key: str):
            nonlocal doctors
            directory = {
                "nephrology": [
                    {"name": "Dr. A. Sharma", "role": "Senior Nephrologist", "hospital": "City Kidney Care Center", "city": "Mumbai"},
                    {"name": "Dr. R. Menon", "role": "Consultant Nephrologist", "hospital": "Metro Multispeciality Hospital", "city": "Delhi"},
                ],
                "cardiology": [
                    {"name": "Dr. S. Verma", "role": "Interventional Cardiologist", "hospital": "Heart Institute", "city": "Bengaluru"},
                    {"name": "Dr. P. Rao", "role": "Cardiology Consultant", "hospital": "Global Cardiac Care", "city": "Chennai"},
                ],
                "hepatology": [
                    {"name": "Dr. M. Iyer", "role": "Hepatologist", "hospital": "Liver & Digestive Clinic", "city": "Hyderabad"},
                    {"name": "Dr. K. Das", "role": "Gastroenterologist", "hospital": "Gastro Health Center", "city": "Kolkata"},
                ],
                "pulmonology": [
                    {"name": "Dr. N. Gupta", "role": "Pulmonologist", "hospital": "Chest & Lung Institute", "city": "Pune"},
                    {"name": "Dr. F. Ahmed", "role": "Respiratory Specialist", "hospital": "Respicare Hospital", "city": "Jaipur"},
                ],
                "endocrinology": [
                    {"name": "Dr. R. Singh", "role": "Endocrinologist", "hospital": "Diabetes & Hormone Clinic", "city": "Ahmedabad"},
                    {"name": "Dr. L. Nair", "role": "Diabetologist", "hospital": "Metabolic Care Center", "city": "Kochi"},
                ],
                "general": [
                    {"name": "Dr. T. Mishra", "role": "General Physician", "hospital": "Family Health Clinic", "city": "Lucknow"},
                    {"name": "Dr. J. Kulkarni", "role": "Internal Medicine Specialist", "hospital": "City General Hospital", "city": "Nagpur"},
                ],
            }
            doctors = directory.get(specialty_key, directory["general"])

        if any(word in description for word in ["kidney", "renal", "creatinine", "urea", "ckd"]):
            recommendation = "Consult a nephrologist (kidney specialist)."
            set_doctors("nephrology")
        elif any(word in description for word in ["heart", "cardiac", "chest pain", "angina", "palpitation"]):
            recommendation = "Consult a cardiologist (heart specialist)."
            set_doctors("cardiology")
        elif any(word in description for word in ["liver", "hepatic", "bilirubin", "jaundice"]):
            recommendation = "Consult a hepatologist or gastroenterologist (liver specialist)."
            set_doctors("hepatology")
        elif any(word in description for word in ["lung", "respiratory", "cough", "breath", "wheezing", "smoking", "cancer"]):
            recommendation = "Consult a pulmonologist or oncologist (lung specialist)."
            set_doctors("pulmonology")
        elif any(word in description for word in ["diabetes", "sugar", "glucose", "insulin"]):
            recommendation = "Consult an endocrinologist (diabetes specialist)."
            set_doctors("endocrinology")
        else:
            recommendation = "Consult a general physician or internal medicine specialist for further evaluation."
            set_doctors("general")

        notes = (
            "This is a suggestion based on your description and is not a medical diagnosis. "
            "Always seek emergency care for severe or sudden symptoms."
        )

    return render_template(
        "contact_doctor.html",
        recommendation=recommendation,
        notes=notes,
        doctors=doctors,
    )


# DIABETES

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    age = float(request.form["age"])
    glucose = float(request.form["glucose"])
    bmi = float(request.form["bmi"])
    dpf = float(request.form["dpf"])
    bp = float(request.form["bp"])

    data = np.array([[age, glucose, bmi, dpf, bp]])

    pred = diabetes_model.predict(data)
    prob = diabetes_model.predict_proba(data)

    risk = prob[0][1]*100

    if pred[0] == 1:
        result = "⚠️ High Risk of Diabetes"
    else:
        result = "✅ Low Diabetes Risk"

    confidence = f"Risk Probability: {risk:.2f}%"

    return render_template("diabetes.html",
                           result_text=result,
                           confidence_text=confidence)



# HEART

@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    age = float(request.form["age"])
    cp = float(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    thalach = float(request.form["thalach"])

    yn_map = {"yes":1,"no":0}

    exang = yn_map[request.form["exang"].lower()]

    data = [[age, cp, trestbps, chol, thalach, exang]]

    pred = heart_model.predict(data)
    prob = heart_model.predict_proba(data)

    risk = prob[0][1] * 100

    if pred[0] == 1:
        result = "⚠️ High Risk of Heart Disease"
    else:
        result = "✅ Low Heart Disease Risk"

    confidence = f"Risk Probability: {risk:.2f}%"

    return render_template("heart.html",
                           result_text=result,
                           confidence_text=confidence)



# KIDNEY 

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():

    age = float(request.form["age"])
    bp = float(request.form["bp"])
    bgr = float(request.form["bgr"])
    bu = float(request.form["bu"])
    sc = float(request.form["sc"])

    data = np.array([[age, bp, bgr, bu, sc]])

    pred = kidney_model.predict(data)
    prob = kidney_model.predict_proba(data)

    risk = prob[0][1]*100

    if pred[0] == 0:
        result = "⚠️ High Risk of Kidney Disease"
    else:
        result = "✅ No Kidney Disease Detected"

    confidence = f"Risk Probability: {risk:.2f}%"

    return render_template("kidney.html",
                           result_text=result,
                           confidence_text=confidence)



# LIVER

@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/predict_liver", methods=["POST"])
def predict_liver():

    tb = float(request.form["tb"])
    alt = float(request.form["alt"])
    ast = float(request.form["ast"])
    alb = float(request.form["alb"])

    data = np.array([[tb, alt, ast, alb]])

    pred = liver_model.predict(data)
    prob = liver_model.predict_proba(data)

    risk = prob[0][1]*100

    if pred[0] == 1:
        result = "⚠️ High Risk of Liver Disease"
    else:
        result = "✅ Low Liver Disease Risk"

    confidence = f"Risk Probability: {risk:.2f}%"

    return render_template("liver.html",
                           result_text=result,
                           confidence_text=confidence)



# LUNG

@app.route("/lung_cancer")
def lung_cancer():
    return render_template("lung_cancer.html")


@app.route("/predict_lung_cancer", methods=["POST"])
def predict_lung_cancer():

    age = float(request.form["age"])
    smoking = float(request.form["smoking"])
    coughing = float(request.form["coughing"])
    breath = float(request.form["shortness_of_breath"])
    chest = float(request.form["chest_pain"])
    wheezing = float(request.form["wheezing"])

    data = np.array([[age, smoking, coughing, breath, chest, wheezing]])

    pred = lung_model.predict(data)
    prob = lung_model.predict_proba(data)

    risk = prob[0][1]*100

    if pred[0] == 1:
        result = "⚠️ High Risk of Lung Disease"
    else:
        result = "✅ Low Lung Disease Risk"

    confidence = f"Risk Probability: {risk:.2f}%"

    return render_template("lung_cancer.html",
                           result_text=result,
                           confidence_text=confidence)

if __name__ == "__main__":
    app.run(debug=True)

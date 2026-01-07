import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure GenAI only if key exists
if API_KEY:
    genai.configure(api_key=API_KEY)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

def extract_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

@app.route("/analyze", methods=["POST"])
def analyze():
    print("Analyze API called")

    if "resume" not in request.files:
        return jsonify({"score": "0"})

    resume = request.files["resume"]
    job_desc = request.form.get("job_description", "")

    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
    resume.save(resume_path)

    resume_text = extract_pdf_text(resume_path)

    # 🔹 Fallback ATS score logic (works even without AI)
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_desc.lower().split())

    if len(jd_words) == 0:
        fallback_score = 0
    else:
        match = resume_words.intersection(jd_words)
        fallback_score = int((len(match) / len(jd_words)) * 100)

    # 🔹 Try AI score
    if API_KEY:
        try:
            prompt = f"""
            Compare resume and job description.
            Give ATS score out of 100.
            Return only number.

            Resume:
            {resume_text}

            Job Description:
            {job_desc}
            """
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            ai_score = response.text.strip()
            return jsonify({"score": ai_score})
        except Exception as e:
            print("AI error:", e)

    return jsonify({"score": str(fallback_score)})

if __name__ == "__main__":
    app.run(debug=True)

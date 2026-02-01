import os
import json
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'CHANGE_THIS_SECRET'

# Data directories and files
KNOWN_DIR = 'known_faces'
VOTERS_DB = 'voters.json'
STAFF_DB = 'staff.json'
VOTES_DB = 'votes.json'
CANDIDATES_DB = 'candidates.json'
CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Create folders if they don't exist
os.makedirs(KNOWN_DIR, exist_ok=True)

# Initialize JSON files if missing or empty
def initialize_file(file_path, default_data):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as fp:
            json.dump(default_data, fp)

initialize_file(VOTERS_DB, {})
initialize_file(STAFF_DB, {})
initialize_file(VOTES_DB, {})
initialize_file(CANDIDATES_DB, {
    "1": "Sai pranay  (*)",
    "2": "Vinay  (#)",
    "3": "Nanda Kishor  ($)"
})

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(CASCADE)

# --- Utility functions ---

def load_json(filepath):
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return {}
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    voters = load_json(VOTERS_DB)
    for vid in voters:
        img_path = os.path.join(KNOWN_DIR, f"{vid}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(int(vid))
    if faces:
        recognizer.train(faces, np.array(labels))
    return recognizer

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(rects):
        x, y, w, h = rects[0]
        return gray[y:y+h, x:x+w]
    return None

# --- Routes ---

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        staff = load_json(STAFF_DB)
        if uname in staff and check_password_hash(staff[uname], pwd):
            session['user'] = uname
            return redirect(url_for('dashboard'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        staff = load_json(STAFF_DB)
        if uname in staff:
            return "Username exists", 400
        staff[uname] = generate_password_hash(pwd)
        with open(STAFF_DB, 'w') as f:
            json.dump(staff, f)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/register_voter', methods=['GET', 'POST'])
def register_voter():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            vid = request.form['voter_id']
            name = request.form['name']
            addr = request.form['address']
            dob = request.form['dob']
            face_data = request.form.get('face_data')

            if not face_data:
                return "Face data is missing", 400

            b64 = face_data.split(',')[1]
            img_bytes = cv2.imdecode(
                np.frombuffer(base64.b64decode(b64), np.uint8),
                cv2.IMREAD_COLOR
            )

            face = detect_face(img_bytes)
            if face is None:
                return "No face detected. Make sure your face is clearly visible.", 400

            cv2.imwrite(os.path.join(KNOWN_DIR, f"{vid}.jpg"), face)

            voters = load_json(VOTERS_DB)
            if vid in voters:
                return "Voter ID already registered", 400

            voters[vid] = {'name': name, 'address': addr, 'dob': dob}
            with open(VOTERS_DB, 'w') as f:
                json.dump(voters, f)

            return f"Voter {name} (ID: {vid}) successfully registered! <a href='/dashboard'>Back to Dashboard</a>"

        except Exception as e:
            return f"Error during registration: {e}", 400

    return render_template('register_voter.html')

@app.route('/vote', methods=['GET', 'POST'])
def vote():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            if 'face_data' in request.form:
                face_data = request.form['face_data']
                if not face_data:
                    return "No face_data provided", 400

                b64 = face_data.split(',')[1]
                img_data = base64.b64decode(b64)
                img_array = np.frombuffer(img_data, np.uint8)
                img_bytes = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bytes is None:
                    return "Failed to decode image", 400

                face = detect_face(img_bytes)
                if face is None:
                    return "No face detected", 400

                recognizer = train_recognizer()
                if recognizer.empty():
                    return "Recognizer has no data to predict", 400

                label, conf = recognizer.predict(face)
                print(f"Predicted label: {label}, confidence: {conf}")

                if conf < 50:
                    voters = load_json(VOTERS_DB)
                    voter = voters.get(str(label))
                    if voter is None:
                        return "Voter not found in database", 404

                    votes = load_json(VOTES_DB)
                    if str(label) in votes:
                        return "You have already voted.", 403

                    candidates = load_json(CANDIDATES_DB)
                    return render_template('cast_vote.html',
                                           voter=voter,
                                           voter_id=label,
                                           candidates=candidates)

                return "Voter not recognized", 404

            elif 'candidate_id' in request.form:
                voter_id = str(request.form.get('voter_id', ''))
                candidate_id = request.form['candidate_id']

                if not voter_id or not candidate_id:
                    return "Voter ID and Candidate ID are required", 400

                votes = load_json(VOTES_DB)
                if voter_id in votes:
                    return "You have already voted.", 403

                votes[voter_id] = candidate_id
                with open(VOTES_DB, 'w') as f:
                    json.dump(votes, f)

                # Redirect to thank you page instead of results page
                return redirect(url_for('thank_you'))

            else:
                return "Invalid POST data", 400

        except Exception as e:
            return f"Error: {e}", 400

    return render_template('vote.html')

@app.route('/thank_you')
def thank_you():
    return """
    <h2>Thank You for Voting!</h2>
    <p>Your vote has been successfully recorded.</p>
    <a href='/dashboard'>Back to Dashboard</a>
    """

@app.route('/results')
def results():
    # Show overall election results, no voter ID required
    candidates = load_json(CANDIDATES_DB)
    votes = load_json(VOTES_DB)

    results = {name: 0 for name in candidates.values()}
    for candidate_id in votes.values():
        candidate_name = candidates.get(candidate_id, "Unknown")
        if candidate_name in results:
            results[candidate_name] += 1

    return render_template('result.html', results=results)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

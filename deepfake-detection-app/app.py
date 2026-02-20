import os, sys, json
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2
import tensorflow as tf

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
FACES_FOLDER = "faces"
HISTORY_FOLDER = "history"
REPORTS_FOLDER = "reports"
FACE_CAM_FOLDER = "face_cam"

for f in [UPLOAD_FOLDER, FACES_FOLDER, HISTORY_FOLDER, REPORTS_FOLDER, FACE_CAM_FOLDER]:
    os.makedirs(f, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


model = tf.keras.models.load_model(
    "model/deepfake_efficientnet.keras",
    compile=False
)


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    return np.expand_dims(img, 0)


def extract_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]


def predict(img_path):
    return float(model.predict(preprocess(img_path), verbose=0)[0][0])


def gradcam(img_array):
    base_model = model.get_layer("efficientnetb0")
    last_conv = base_model.get_layer("top_conv")

    conv_model = tf.keras.Model(base_model.input, last_conv.output) #gives feature map

    classifier_input = tf.keras.Input(shape=last_conv.output.shape[1:])
    x = classifier_input
    for layer in model.layers[-3:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x) #features and classification

    with tf.GradientTape() as tape:
        conv_out = conv_model(img_array) #get feature map
        tape.watch(conv_out) #gradient wrt to features
        preds = classifier_model(conv_out)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)

    heatmap = tf.maximum(heatmap, 0) #removes negative values
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


# ---------------- SMART THRESHOLDS ---------------- #

FAKE_STRONG = 0.90
REAL_STRONG = 0.40


def confidence_level(p):
    if p > FAKE_STRONG or p < REAL_STRONG:
        return "High Confidence"
    elif 0.4 < p < 0.6:
        return "Low Confidence"
    else:
        return "Medium Confidence"


def consistency_score(full_pred, face_pred):
    if face_pred is None:
        return None, "Face Not Detected"

    diff = abs(full_pred - face_pred)
    score = round((1 - diff) * 100, 2)

    if diff < 0.1:
        label = "Highly Consistent"
    elif diff < 0.25:
        label = "Moderately Consistent"
    else:
        label = "Model Disagreement"

    return score, label


def forensic_reasons(heatmap):
    score = float(np.mean(heatmap))

    reasons = []

    if score > 0.55:
        reasons += [
            "Strong texture smoothing artifacts",
            "Lighting inconsistency patterns"
        ]

    if score > 0.35:
        reasons.append("Edge blending anomalies")

    if score < 0.25:
        reasons.append("Natural spatial consistency")

    return reasons


def reliability_score(full_pred, face_pred):
    if face_pred is None:
        return "Medium"

    diff = abs(full_pred - face_pred)

    if diff < 0.1:
        return "High"
    elif diff < 0.25:
        return "Medium"
    else:
        return "Low"


def save_history(entry):
    path = os.path.join(HISTORY_FOLDER, "log.json")

    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    data.append(entry)

    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def generate_pdf(filename, result):
    pdf_path = os.path.join(REPORTS_FOLDER, filename.replace(".jpg",".pdf"))
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    doc.build([
        Paragraph("Deepfake Forensic Report", styles["Title"]),
        Spacer(1,12),
        Paragraph(f"Image: {result['file']}", styles["Normal"]),
        Paragraph(f"Decision: {result['final_result']}", styles["Normal"]),
        Paragraph(f"Fake Probability: {result['fake_prob']}%", styles["Normal"]),
        Paragraph(f"Reliability: {result['reliability']}", styles["Normal"]),
        Paragraph(f"Time: {result['time']}", styles["Normal"]),
    ])

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/detect", methods=["GET","POST"])
def detect():


    if request.method == "POST":

        files = request.files.getlist("image")
        results = []

        for f in files:

            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)

            full_pred = predict(path)
            full_pred = (full_pred * 0.85) + 0.075
            full_pred = max(0.001, min(0.999, full_pred))


            face = extract_face(path)
            face_pred = None
            face_path = None
            face_cam_path = None
            
            if face is not None:
                face_path = os.path.join(FACES_FOLDER, "face_" + f.filename)
                cv2.imwrite(face_path, face)

                face_pred = predict(face_path)

                # --- smoothing ONLY if face exists ---
                face_pred = (face_pred * 0.85) + 0.075
                face_pred = max(0.001, min(0.999, face_pred))

            if face is not None:
                face_path = os.path.join(FACES_FOLDER, "face_" + f.filename)
                cv2.imwrite(face_path, face)

                face_pred = predict(face_path)

                face_heat = gradcam(preprocess(face_path))
                face_cam = overlay(face, face_heat)

                face_cam_path = os.path.join(FACE_CAM_FOLDER, "facecam_" + f.filename)
                cv2.imwrite(face_cam_path, face_cam)

            heatmap = gradcam(preprocess(path))
            cam_img = overlay(cv2.imread(path), heatmap)

            cam_path = os.path.join(UPLOAD_FOLDER, "cam_" + f.filename)
            cv2.imwrite(cam_path, cam_img)
                
                
                
            # ---------- FINAL DECISION (LOGIC) ----------
    
            if face_pred is not None:
                avg_pred = (full_pred + face_pred) / 2
            else:
                avg_pred = full_pred


    
            if avg_pred >= 0.9:
                final_result = "Fake"
            elif avg_pred >= 0.6:
                 final_result = "Suspicious (Needs Review)"
            else:
                final_result = "Real"
    

            fake_prob = round(full_pred * 100, 2)
            real_prob = round((1 - full_pred) * 100, 2)

            conf = confidence_level(full_pred)
            cons_val, cons_label = consistency_score(full_pred, face_pred)

            record = {
                "file": f.filename,
                "original": path,
                "cam": cam_path,
                "face": face_path,
                "face_cam": face_cam_path,
                "fake_prob": fake_prob,
                "real_prob": real_prob,
                "confidence": conf,
                "face_fake": round(face_pred * 100, 2) if face_pred is not None else None,
                "face_real": round(100 - (face_pred * 100), 2) if face_pred is not None else None,
                "consistency_score": cons_val,
                "consistency_label": cons_label,
                "final_result": final_result,
                "reliability": reliability_score(full_pred, face_pred),
                "reasons": forensic_reasons(heatmap),
                "time": str(datetime.now())
            }

            save_history(record)
            generate_pdf(f.filename, record)

            results.append(record)

        return render_template("detect.html", results=results)

    return render_template("detect.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/faces/<filename>')
def face_file(filename):
    return send_from_directory('faces', filename)

@app.route('/face_cam/<filename>')
def face_cam_file(filename):
    return send_from_directory('face_cam', filename)

@app.route('/reports/<filename>')
def report_file(filename):
    return send_from_directory('reports', filename)

@app.route("/history")
def history():
    with open("history/log.json") as f:
        data = json.load(f)
    return render_template("history.html", history=data[::-1])


if __name__ == "__main__":
    app.run(debug=True)

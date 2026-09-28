from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from datetime import datetime
import uuid
import cv2
import mediapipe as mp
import joblib
import numpy as np
from sklearn.preprocessing import normalize
import base64
import os
import logging



ADAPTIVE_DIR = os.path.join(BASE_DIR , "adaptive")
CANDIDATES_PATH = os.path.join(ADAPTIVE_DIR , "adaptive.json")

os.makedir(ADAPTIVE_DIR , exist_ok = True)

if not os.path.exists(CANDIDATES_PATH):
    with open(CANDIDATES_PATH, "W") as f:
        json.dump([] , f)

def save_candidates(features , predicted_label , confidence , reason):
    try:
        with open(CANDIDATES_PATH , "r") as f:
            candidates = json.load(f)

        candidate = {
            "id": str(uuid.uuid4()),
            "timestamp":datetime.now().isoformat(),
            "predicted_label": predicted_label ,
            "confidence": round(float(confidence) , 4) ,
            "label":"",
            "review_status":"PENDING",
            "reason":reason,
            "features":features.tolist()
        } 
        candidates.append(candidate)

        with open(CANDIDATES_PATH , "w") as f:
            json.dump(candidates , f , indent = 2)
        
        return candidate
    
    except Exception as e:
        log.error(f"candidate save error: {e}")
        return home

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


CONFIDENCE_THRESHOLD     = 0.65   
MIN_DETECTION_CONFIDENCE = 0.6    
MIN_TRACKING_CONFIDENCE  = 0.5

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

log.info(f"Loading model  : {MODEL_PATH}")
log.info(f"Loading encoder: {ENCODER_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"model.pkl not found at {MODEL_PATH}\n"
        "Run train_model.py first!"
    )
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"label_encoder.pkl not found at {ENCODER_PATH}\n"
        "Run train_model.py first!"
    )

model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
CLASS_NAMES = list(encoder.classes_)
log.info(f"Model loaded. Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
KEY_PAIRS = [
    (0,  4),   
    (0,  8),   
    (0, 12),   
    (0, 16),   
    (0, 20),   
    (4,  8),   
    (8, 12),   
    (5,  9),   
]

def engineer_single(raw_xy: list[float]) -> np.ndarray:
    row = np.array(raw_xy, dtype=np.float32)


    bx, by = row[0], row[1]
    for j in range(0, len(row), 2):
        row[j]   -= bx
        row[j+1] -= by

    row = normalize(row.reshape(1, -1), norm="l2").flatten()

    dists = []
    for (a, b) in KEY_PAIRS:
        ax, ay = row[a*2], row[a*2+1]
        bx, by = row[b*2], row[b*2+1]
        dists.append(float(np.sqrt((ax - bx)**2 + (ay - by)**2)))

    return np.concatenate([row, dists])          # shape (50,)


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("image", "")
        if not data:
            return jsonify({"label": "No Image", "confidence": 0.0})

        header, encoded = data.split(",", 1) if "," in data else ("", data)
        img_bytes = base64.b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"label": "Bad Frame", "confidence": 0.0})

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return jsonify({"label": "No Hand", "confidence": 0.0})

        hand_lms = result.multi_hand_landmarks[0]

        raw_xy = []
        for lm in hand_lms.landmark:
            raw_xy.append(lm.x)
            raw_xy.append(lm.y)

        if len(raw_xy) != 42:
            return jsonify({"label": "Bad Landmarks", "confidence": 0.0})

        features = engineer_single(raw_xy)          # shape (50,)

        probs    = model.predict_proba([features])[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])

        predicted_label = encoder.inverse_transform([best_idx])[0]


        if best_prob < CONFIDENCE_THRESHOLD:
            save_candidate(
                features , 
                predicted_label,
                best_prob,
                "LOW_CONFIDENCE"
            )
        if best_prob >= CONFIDENCE_THRESHOLD:
            return jsonify({
                "label":      predicted_label,
                "confidence": round(best_prob, 4),
            })
        else:
            return jsonify({
                "label":      "Unknown",
                "confidence": round(best_prob, 4),
            })

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"label": "Error", "confidence": 0.0})


@app.route("/health", methods=["GET"])
def health():
    """Quick sanity-check endpoint."""
    return jsonify({
        "status":      "ok",
        "num_classes": len(CLASS_NAMES),
        "classes":     CLASS_NAMES,
        "threshold":   CONFIDENCE_THRESHOLD,
        "model_type":  type(model).__name__,
    })

@app.route("/adaptive/candidates" , methods=["GET"])
def get_candidates():
    with open(CANDIDATES_PATH , "r") as f:
        candidates = json.load(f)
    
    return jsonify(candidates)

@app.route("/adaptive/candidates/<canddidate_id>/approve" , methods = ["POST"])
def approve_candidate(candidate_id):
    data = request.json or {}
    label = data.get("label")

    if not label:
        return jsonify({"error": "Label is required"}),400

    with open(CANDIDATES_PATH , "r") as f:
        candidate = json.load(f)
    
    found = None

    for candidate in candidates:
        if candidate["id"] == canddidate_id:
            candidate["label"] = label
            candidate["review_status"] = "APPROVED"
            APPROVED_PATH = os.path.join(ADAPTIVE_DIR , "approve.json")
            if not os.path.exists(APPROVED_PATH):
                with open(APPROVED_PATH , "w") as f:
                    approved = json.load(f)
            approved.append({
                "id": candidate(["id"]),
                "label":label,
                "features" : candidate["features"],
                "timestamp": candidate["timestamp"]
            })
            with open(APPROVED_PATH , "W") as f:
                json.dump(approved , f , indent = 2)
            found = candidate
            break
        
        if not found:
            return jsonify({"error": "candidate not found"}), 404
        
        with open(CANDIDATES_PATH , "w") as f:
            json.dump(candidates , f  , indent = 2)
        
        return jsonify({
            "message" : "candidate approved" , 
            "candidate" : found
        })

@app.route("/classes", methods=["GET"])
def classes():
    """Return all mudra class names — useful for the frontend."""
    return jsonify({"classes": CLASS_NAMES})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log.info(f"Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
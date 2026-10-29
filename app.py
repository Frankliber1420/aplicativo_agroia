import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms, models
import joblib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)
app.secret_key = "agroai_secret_key_fitosanitario_palmas"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agroai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class HistoryRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)

with app.app_context():
    db.create_all()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["MANCHAS FOLIARES", "MARCHITEZ AVANZADA", "MARCHITEZ SORPRESIVA", "SANA"]
NUM_CLASSES = len(CLASSES)

MODELO_SEG_PATH = os.path.join("weights", "best.pt")
MODEL_DETECTOR_PATH = os.path.join("weights", "detector_palma_vs_nopalma.pkl")
MODEL_EFF_PATH = os.path.join("weights", "best_effb0_palma_model.pth")

model_seg = None
detector_dominio = None
model_eff = None

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_all_models():
    global model_seg, detector_dominio, model_eff
    
    if os.path.exists(MODELO_SEG_PATH) and YOLO is not None:
        try:
            model_seg = YOLO(MODELO_SEG_PATH)
            print(f"[*] Modelo YOLO cargado desde {MODELO_SEG_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar YOLO: {e}")

    if os.path.exists(MODEL_DETECTOR_PATH):
        try:
            detector_dominio = joblib.load(MODEL_DETECTOR_PATH)
            print(f"[*] Detector Anti-OOD cargado desde {MODEL_DETECTOR_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar detector: {e}")

    if os.path.exists(MODEL_EFF_PATH):
        try:
            model_eff = models.efficientnet_b0(weights=None)
            in_features = model_eff.classifier[1].in_features
            model_eff.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, NUM_CLASSES)
            )
            model_eff.load_state_dict(torch.load(MODEL_EFF_PATH, map_location=DEVICE))
            model_eff.to(DEVICE)
            model_eff.eval()
            print(f"[*] Clasificador EfficientNet-B0 cargado desde {MODEL_EFF_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar EfficientNet-B0: {e}")

load_all_models()

RECOMMENDATIONS = {
    "MANCHAS FOLIARES": {
        "severidad": "Afectación Moderada",
        "color_badge": "bg-warning text-dark",
        "acciones": [
            "Realizar poda sanitaria de las hojas con síntomas avanzados para reducir inóculo.",
            "Aplicar fungicida protectante/sistémico a base de triazoles o estrobilurinas.",
            "Verificar y mejorar el sistema de drenaje para prevenir acumulación de humedad.",
            "Monitorear la parcela cada 8 días para evitar diseminación."
        ]
    },
    "MARCHITEZ AVANZADA": {
        "severidad": "Severidad Alta / Crítica",
        "color_badge": "bg-danger text-white",
        "acciones": [
            "Aislar inmediatamente la palma para evitar transmisión radicular o por vectores.",
            "Evaluar erradicación de la palma afectada siguiendo protocolos del ICA.",
            "Desinfectar herramientas de poda con hipoclorito de sodio al 10% entre plantas.",
            "Tomar muestras de raíz/suelo para confirmación fitopatológica en laboratorio."
        ]
    },
    "MARCHITEZ SORPRESIVA": {
        "severidad": "Severidad Alta / Cuarentenaria",
        "color_badge": "bg-danger text-white",
        "acciones": [
            "Inspección focalizada del vector chinche de la palma (Lincus sp.).",
            "Aplicar control integrado de vectores en el lote y lotes vecinos.",
            "Erradicación inmediata de la palma afectada si hay marchitamiento acelerado.",
            "Reportar el foco al departamento técnico agronómico de la zona."
        ]
    },
    "SANA": {
        "severidad": "Estado Óptimo / Sin Síntomas",
        "color_badge": "bg-success text-white",
        "acciones": [
            "Mantener el plan nutricional y programa de fertilización de rutina.",
            "Continuar con los plateos y limpiezas programadas del cultivo.",
            "Realizar monitoreo preventivo quincenal en hojas del estrato medio."
        ]
    }
}

def aplicar_bokeh(img_pil):
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_np.shape
    
    try:
        if model_seg is not None:
            results = model_seg(img_np, verbose=False)[0]
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Prioridad 1: Usar máscaras de segmentación exactas de YOLO
            if hasattr(results, "masks") and results.masks is not None and len(results.masks) > 0:
                for m in results.masks.data:
                    m_np = m.cpu().numpy().astype(np.float32)
                    m_resized = cv2.resize(m_np, (w, h))
                    mask = np.maximum(mask, m_resized)
            
            # Prioridad 2: Respaldo con cajas delimitadoras si no hay máscaras
            elif hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes.xyxy.cpu().numpy():
                    xmin, ymin, xmax, ymax = map(int, box[:4])
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmax)
                    ymax = min(h, ymax)
                    mask[ymin:ymax, xmin:xmax] = 1.0
            
            if np.sum(mask) > 0:
                fondo_borroso = cv2.GaussianBlur(img_np, (99, 99), 0)
                mask_soft = cv2.GaussianBlur(mask, (31, 31), 0)
                mask_3ch = np.repeat(mask_soft[:, :, np.newaxis], 3, axis=2)
                resultado = (img_np * mask_3ch + fondo_borroso * (1.0 - mask_3ch)).astype(np.uint8)
                return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
                
        return img_pil
    except Exception as e:
        print(f"[!] Error en Bokeh: {e}")
        return img_pil

def extraer_descriptores_fisionomicos(img_pil):
    img_np = np.array(img_pil.resize((128, 128)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    edges = cv2.Canny(gray_eq, 30, 130)
    densidad_bordes = np.sum(edges > 0) / (128.0 * 128.0)

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist_angle, _ = np.histogram(angle, bins=12, range=(0, 360), weights=mag)
    hist_angle = hist_angle / (np.sum(hist_angle) + 1e-6)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))

    return np.hstack([densidad_bordes, hist_angle, mean_hsv, std_hsv])

def generate_confidence_chart(probabilities, classes, predicted_class, confidence_val):
    fig, ax = plt.subplots(figsize=(5, 4))
    y_pos = np.arange(len(classes))
    probs = [p * 100 for p in probabilities]
    
    ax.barh(y_pos, probs, color="#10b981", height=0.45)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9, fontweight="bold", color="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("Confianza (%)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_title(f"Predicción: {predicted_class} ({confidence_val:.1f}%)", fontsize=10, fontweight="bold")
    
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_to_base64(pil_img):
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session["user"] = user.email
        session["user_name"] = user.name
        return jsonify({"success": True, "user_name": user.name})
    return jsonify({"success": False, "error": "Correo o contraseña incorrectos"}), 401

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"success": False, "error": "Todos los campos son obligatorios"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "error": "El correo ya se encuentra registrado"}), 400
    
    new_user = User(name=name, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    
    session["user"] = new_user.email
    session["user_name"] = new_user.name
    return jsonify({"success": True, "user_name": new_user.name})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/")
def index():
    user_name = session.get("user_name", None)
    return render_template("index.html", user_name=user_name)

@app.route("/history", methods=["GET"])
def get_history():
    if "user" not in session:
        return jsonify({"success": True, "records": []})
    
    user_email = session["user"]
    records = HistoryRecord.query.filter_by(user_email=user_email).order_by(HistoryRecord.id.desc()).all()
    
    records_list = [{
        "timestamp": r.timestamp,
        "filename": r.filename,
        "diagnosis": r.diagnosis,
        "confidence": r.confidence
    } for r in records]
    
    return jsonify({"success": True, "records": records_list})

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Acceso denegado. Inicia sesión."}), 401
    
    try:
        pil_img = None
        filename_str = "imagen_sistema.jpg"
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename_str = file.filename
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif request.is_json and "image_path" in request.json:
            img_path = request.json["image_path"].lstrip("/")
            if os.path.exists(img_path):
                filename_str = os.path.basename(img_path)
                pil_img = Image.open(img_path).convert("RGB")
                
        if pil_img is None:
            return jsonify({"error": "No se seleccionó ni cargó ninguna imagen válida."}), 400
        
        bokeh_img = aplicar_bokeh(pil_img)
        
        if detector_dominio is not None:
            feat = extraer_descriptores_fisionomicos(bokeh_img).reshape(1, -1)
            es_palma_valida = detector_dominio.predict(feat)[0]
            if es_palma_valida == -1:
                return jsonify({
                    "success": False,
                    "error": "La imagen cargada no corresponde a una palma de aceite o no presenta la estructura anatómica foliar válida."
                }), 400

        if model_eff is not None:
            tensor_img = val_transforms(bokeh_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model_eff(tensor_img)
                probs = F.softmax(outputs, dim=1)
                probabilities = probs.cpu().numpy()[0].tolist()
        else:
            probabilities = [0.25, 0.25, 0.25, 0.25]
            
        pred_idx = int(np.argmax(probabilities))
        predicted_class = CLASSES[pred_idx]
        confidence = probabilities[pred_idx] * 100
        
        chart_b64 = generate_confidence_chart(probabilities, CLASSES, predicted_class, confidence)
        img_entrada_b64 = pil_to_base64(bokeh_img)
        
        if model_seg is not None:
            res_yolo = model_seg(pil_img, verbose=False)[0]
            gradcam_bgr = res_yolo.plot()
            gradcam_rgb = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)
            gradcam_b64 = pil_to_base64(gradcam_rgb)
        else:
            gradcam_b64 = pil_to_base64(bokeh_img)
        
        recom_data = RECOMMENDATIONS.get(predicted_class, RECOMMENDATIONS["SANA"])
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_record = HistoryRecord(
            user_email=session["user"],
            timestamp=current_time,
            filename=filename_str,
            diagnosis=predicted_class,
            confidence=f"{confidence:.1f}%"
        )
        db.session.add(new_record)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.1f}%",
            "severidad": recom_data["severidad"],
            "color_badge": recom_data["color_badge"],
            "acciones": recom_data["acciones"],
            "img_entrada": f"data:image/png;base64,{img_entrada_b64}",
            "gradcam": f"data:image/png;base64,{gradcam_b64}",
            "chart": f"data:image/png;base64,{chart_b64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)


from flask import Blueprint, request, jsonify
from langdetect import detect
import cv2
import numpy as np
from PIL import Image

from app.config import MODEL_PATH
from app.services.model_service import (
    SESSIONS,
    QUESTIONS,
    predict_en,
    build_recommendations,
    norm_lang,
    translate,
    cv_model,
    class_names,
    check_timeout,
    TR_LABELS,
    TR_SEVERITY,
    postprocess_cv_probs,
    score_text_description,
    fuse_cv_and_text,
    normalize_short_answer,             
    expand_short_answer_for_question,   
)

from app.utils.helpers import (
    error_response,
    extract_face,
    calc_brightness,
    calc_sharpness,
)

main_bp = Blueprint("main", __name__)


# ==========================
# ROOT
# ==========================
@main_bp.route("/", methods=["GET"])
def home():
    return jsonify({"ok": True, "status": "running", "model": MODEL_PATH})


# ==========================
# START SESSION
# ==========================


@main_bp.route("/start_session", methods=["POST"])
def start_session():
    data = request.get_json() or {}
    text = data.get("text", "")

    try:
        detected = detect(text or "Merhaba")
        lang = norm_lang(detected)
    except Exception:
        lang = "tr"

    if lang not in QUESTIONS:
        lang = "tr"

    session_id = f"sess_{len(SESSIONS) + 1}"
    SESSIONS[session_id] = {"step": 0, "answers": [], "lang": lang, "analysis": None}
    first_q = QUESTIONS[lang][0]

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "lang": lang,
        "question": first_q,
        "done": False,
    })


# ==========================
# ANSWER → FINAL ANALYSIS
# ==========================

@main_bp.route("/answer", methods=["POST"])
def answer():
    data = request.get_json() or {}
    session_id = data.get("session_id")

    # ==========================
    # ✅ NEW: Support short answers + optional free text
    # Flutter may send:
    # - answer: "Evet" / "Hayır" / "OK" / long text
    # - short_answer: optional (if you add buttons)
    # - free_text: optional details
    # ==========================
    answer_text = (data.get("answer") or "").strip()
    short_in = (data.get("short_answer") or "").strip()
    free_text = (data.get("free_text") or "").strip()

    if not session_id or session_id not in SESSIONS:
        return error_response("Geçersiz oturum.", "invalid_session", 400)

    sess = SESSIONS[session_id]
    lang = sess["lang"]

    # ===== NEW: Block any reply after session is finished =====
    if sess.get("finished"):
        return jsonify({
            "ok": False,
            "done": True,
            "message": "Bu oturum tamamlandı. Yeni bir değerlendirme için yeniden başlayabilirsiniz.",
            "restart": True
        })

    # ===== timeout check =====
    timeout_state = check_timeout(sess, lang)
    if timeout_state["timeout"]:
        return jsonify({
            "ok": True,
            "done": False,
            "question": timeout_state["msg"]
        })

    q_idx = sess["step"] 
    raw = answer_text or short_in
    short_label = normalize_short_answer(raw, lang)
    expanded = ""
    if short_label != "UNKNOWN":
        expanded = expand_short_answer_for_question(q_idx, short_label, lang)

    if free_text:
        stored_text = (expanded + " " + free_text).strip() if expanded else free_text
    else:
        stored_text = expanded if expanded else raw

    if not stored_text:
        return error_response("Lütfen bir yanıt girin.", "empty_answer", 400)

    from app.services.model_service import is_relevant_to_mental_state, redirect_user

    if not is_relevant_to_mental_state(stored_text, lang):
        return jsonify({
            "ok": False,
            "done": False,
            "message": redirect_user(lang)
        })

    sess["answers"].append(stored_text)
    sess["step"] += 1

    if sess["step"] < len(QUESTIONS[lang]):
        next_q = QUESTIONS[lang][sess["step"]]
        return jsonify({"ok": True, "done": False, "question": next_q})

    combined_text = " ".join(sess["answers"])
    combined_en = translate(combined_text, lang, "en")

    result = predict_en(combined_en)
    sess["analysis"] = result

    sess["finished"] = True

    translated_result = {
        "category": TR_LABELS.get(result["category"], result["category"]),
        "severity": TR_SEVERITY.get(result["severity"], result["severity"]),
        "confidence": result["confidence"],
        "top3": [
            {
                "label": TR_LABELS.get(t["label"], t["label"]),
                "score": t["score"]
            } for t in result["top3"]
        ]
    }

    return jsonify({
        "ok": True,
        "done": True,
        "result": translated_result,
    })


# SUGGESTIONS
@main_bp.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    lang = data.get("lang")
    category = data.get("category")
    severity = data.get("severity")

    if session_id and session_id in SESSIONS and not (category and severity and lang):
        sess = SESSIONS[session_id]
        lang = sess.get("lang", "tr")
        analysis = sess.get("analysis") or {}
        category = analysis.get("category", "Normal")
        severity = analysis.get("severity", "mild")

    lang = lang or "tr"
    category = category or "Normal"
    severity = severity or "mild"

    reverse_labels = {v: k for k, v in TR_LABELS.items()}
    reverse_severity = {v: k for k, v in TR_SEVERITY.items()}

    category_en = reverse_labels.get(category, category)
    severity_en = reverse_severity.get(severity, severity)

    recs = build_recommendations(category_en, severity_en, lang)

    return jsonify({
        "ok": True,
        "lang": lang,
        "category": TR_LABELS.get(category_en, category),
        "severity": TR_SEVERITY.get(severity_en, severity),
        "recommendations": recs,
    })



# IMAGE ANALYSIS (CV + Text Fusion)
def detect_expression(image):
    return "other"


@main_bp.route("/analyze_image", methods=["POST"])
def analyze_image():
    
    try:
        if cv_model is None:
            return error_response(
                "Görüntü analiz servisi şu anda kullanılamıyor.",
                code="model_unavailable",
                http_status=503,
            )

        description = (request.form.get("text") or "").strip()
        if not description:
            return error_response(
                "Lütfen görüntü ile birlikte en az 3 kelimelik bir açıklama yazın.",
                code="missing_text",
                http_status=400,
            )

        word_count = len(description.split())
        if word_count < 3:
            return error_response(
                "Açıklama metni çok kısa. En az 3 kelime yazın.",
                code="text_too_short",
                http_status=400,
            )

        if "image" not in request.files:
            return error_response(
                "Herhangi bir görüntü gönderilmedi.",
                code="invalid_payload",
                http_status=400,
            )

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return error_response(
                "Gönderilen dosya geçerli bir görüntü değil.",
                code="invalid_image",
                http_status=400,
            )

        face_img = extract_face(img)
        if face_img is None:
            return error_response(
                "Görüntüde net bir yüz bulunamadı.",
                code="no_face",
                http_status=400,
            )

        brightness = calc_brightness(face_img)
        if brightness < 35:
            return error_response(
                "Işık çok düşük. Daha iyi aydınlatma ile tekrar deneyin.",
                code="low_light",
                http_status=400,
            )

        if brightness > 230:
            return error_response(
                "Yüzde aşırı ışık tespit edildi. Doğrudan ışığı azaltmayı deneyin.",
                code="high_light",
                http_status=400,
            )

        sharpness = calc_sharpness(face_img)
        if sharpness < 15:
            return error_response(
                "Görüntü net değil. Hareket veya titreme olabilir.",
                code="blurry_image",
                http_status=400,
            )

        # ====== PREPARE IMAGE FOR MODEL ======
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).convert("RGB")
        face_pil = face_pil.resize((224, 224))

        img_array = np.array(face_pil, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ====== CV MODEL PREDICTION ======
        preds = cv_model.predict(img_array)
        raw_probs = preds[0]  # shape: (num_classes,)
        raw_idx = int(np.argmax(raw_probs))
        raw_label = class_names[raw_idx]
        raw_conf = float(np.max(raw_probs))

        cv_probs = postprocess_cv_probs(raw_probs)

        # ====== TEXT SCORING ======
        try:
            detected_lang = detect(description)
            lang = norm_lang(detected_lang)
        except Exception:
            lang = "tr"

        text_scores = score_text_description(description, lang)

        # ====== FUSION ======
        fused_probs = fuse_cv_and_text(cv_probs, text_scores)
        final_idx = int(np.argmax(fused_probs))
        final_label = class_names[final_idx]
        final_conf = float(fused_probs[final_idx])

        top_k = min(3, fused_probs.size)
        top_indices = np.argsort(fused_probs)[-top_k:][::-1]
        top3 = [
            {
                "label": class_names[i],
                "score": float(fused_probs[i]),
            }
            for i in top_indices
        ]

        if final_conf < 0.5:
            return error_response(
                "Model bu görüntü ve açıklamadan emin değil. Daha net fotoğraf ve daha açıklayıcı metin deneyin.",
                code="low_confidence",
                http_status=400,
            )

        return jsonify({
            "ok": True,
            "prediction": final_label,
            "confidence": round(final_conf, 3),
            "top3": top3,
            "meta": {
                "lang": lang,
                "description_used": bool(text_scores.sum() > 0),
                "image_only": {
                    "prediction": raw_label,
                    "confidence": round(raw_conf, 3),
                },
            },
        })

    except Exception as e:
        print(" Hata:", e)
        return error_response(
            "Görüntü analiz edilirken beklenmeyen bir hata oluştu.",
            code="server_error",
            http_status=500,
        )

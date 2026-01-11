from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict
import random
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import time
import re  

from app.config import MODEL_PATH, CV_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=False)
model.eval()

SESSIONS = defaultdict(lambda: {"step": 0, "answers": [], "lang": "tr", "analysis": None})

# ================== Questions (TR only) ==================
QUESTIONS = {
    "tr": [
        "Uykuya dalmakta veya odaklanmakta zorluk çekiyor musunuz?",
        "Hafta boyunca ne sıklıkla üzgün veya endişeli hissediyorsunuz?",
        "Sizi rahatsız eden durumlar veya baskılar var mı?",
        "Ne tür bir destek tercih edersiniz?"
    ]
}

# ================== LABEL TRANSLATION (TR) ==================
TR_LABELS = {
    "Anxiety": "Anksiyete",
    "Depression": "Depresyon",
    "Stress": "Stres",
    "Bipolar": "Bipolar Bozukluk",
    "Severe_Psychiatric_Disorders": "Ağır Psikiyatrik Bozukluklar",
    "Normal": "Normal",
}

# ================== SEVERITY TRANSLATION (TR) ==================
TR_SEVERITY = {
    "mild": "hafif",
    "moderate": "orta",
    "severe": "yüksek"
}

# ================== Session Safety Layer ==================

EMOTION_KEYWORDS = {
    "tr": [
        "kaygı", "kaygılı", "kaygılanıyorum", "kaygılandım",
        "endişe", "endişeliyim", "endişeleniyorum", "endişelendiriyor",
        "gergin", "gerginlik", "geriliyorum", "gerildim",

        "stres", "stresliyim", "stres oldum", "stres yapıyorum",
        "korku", "korkuyorum", "korktum", "korkutuyor",
        "panik", "panikledim", "panikliyorum",

        "üzgün", "üzüntü", "üzüntülüyüm", "üzülüyorum",
        "depresyon", "depresif", "depresyondayım", "moralim bozuk",
        "çökkün", "çökkünlük",

        "ruh", "ruhsal", "ruhen kötü", "ruhsal çöküş",
        "psikolojik", "psikolojim bozuk", "psikolojik baskı",
        "duygusal", "duygusal çöküntü", "duygusal yorgunluk",

        "uyku", "uyuyamıyorum", "uykusuzluk", "uykum kaçtı",
        "uyku sorunu", "kabus", "rahatsız uyku",

        "bunalım", "bunaldım", "bunalıyorum", "boğuluyorum",
        "iç sıkıntısı", "sıkıntı", "sıkılıyorum",

        "intihar", "kendimi öldürmek", "yaşamak istemiyorum",
        "ölüm düşüncesi", "kendime zarar vermek",

        "yalnızlık", "yalnızım", "kimsem yok", "terkedilmiş hissediyorum",
        "desteksiz hissediyorum", "anlaşılmıyorum"
    ]
}


REDIRECT_REPLIES = {
    "tr": [
        "Mesajınız ruh halinizle ilgili görünmüyor. Son zamanlarda nasıl hissediyorsunuz?",
        "Size yardımcı olabilmem için nasıl hissettiğinizi bilmem gerekiyor."
    ]
}

SESSION_TIMEOUT_SECONDS = 180


SHORT_ANSWERS = {
    "tr": {
        "YES": {"evet", "e", "aynen", "olur"},
        "NO": {"hayir", "hayır", "h", "yok", "degil", "değil"},
        "OK": {"ok", "okay", "tamam", "peki", "olabilir"}
    },
    "en": {
        "YES": {"yes", "y", "yeah", "yep", "sure", "true"},
        "NO": {"no", "n", "nope", "false"},
        "OK": {"ok", "okay", "alright"}
    },
    "ar": {
        "YES": {"نعم", "اي", "أجل"},
        "NO": {"لا", "كلا"},
        "OK": {"تمام", "اوكي", "حسنا", "حسنًا"}
    }
}


def normalize_short_answer(text: str, lang: str) -> str:
    """
    Returns: 'YES' | 'NO' | 'OK' | 'UNKNOWN'
    """
    t = (text or "").strip().lower()
    if not t:
        return "UNKNOWN"

    t = re.sub(r"[^\w\s\u0600-\u06FF]", "", t)  
    t = re.sub(r"\s+", " ", t).strip()

    lang = norm_lang(lang)
    bank = SHORT_ANSWERS.get(lang, SHORT_ANSWERS["tr"])

    for label, words in bank.items():
        if t in words:
            return label
    return "UNKNOWN"


def expand_short_answer_for_question(q_idx: int, short_label: str, lang: str) -> str:
    """
    Converts YES/NO/OK into a meaningful TR sentence so the classifier gets signal,
    not just 'Evet'.
    """
    short_label = (short_label or "UNKNOWN").upper()
    _ = norm_lang(lang)  

    if q_idx == 0:
        if short_label == "YES":
            return "Uykuya dalmakta veya odaklanmakta zorluk çekiyorum."
        if short_label == "NO":
            return "Uykuya dalmakta veya odaklanmakta zorluk çekmiyorum."
        if short_label == "OK":
            return "Bazen uykuya dalmakta veya odaklanmakta zorluk çekiyorum."
    elif q_idx == 1:
        if short_label == "YES":
            return "Hafta boyunca sık sık üzgün veya endişeli hissediyorum."
        if short_label == "NO":
            return "Hafta boyunca nadiren üzgün veya endişeli hissediyorum."
        if short_label == "OK":
            return "Bazen üzgün veya endişeli hissediyorum."
    elif q_idx == 2:
        if short_label == "YES":
            return "Beni rahatsız eden durumlar veya baskılar var."
        if short_label == "NO":
            return "Şu anda beni rahatsız eden baskılar çok yok."
        if short_label == "OK":
            return "Bazı baskılar var ama yönetmeye çalışıyorum."
    elif q_idx == 3:
        if short_label in ("YES", "OK"):
            return "Destek almak istiyorum."
        if short_label == "NO":
            return "Şu anda destek istemiyorum."

    return ""


def check_timeout(session, lang):
    last = session.get("last_time")
    now = time.time()

    if last and (now - last > SESSION_TIMEOUT_SECONDS):
        session["step"] = 0
        session["answers"] = []
        session["analysis"] = None
        session["finished"] = False
        return {
            "timeout": True,
            "msg": "Uzun süre yanıt vermediniz. Baştan başlayalım: Şu anda nasıl hissediyorsunuz?"
        }

    session["last_time"] = now
    return {"timeout": False}


def translate(text, src, tgt):
    if src == tgt:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception:
        return text


def predict_en(text_en: str):
    inputs = tokenizer(
        text_en,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()

    pred_idx = int(torch.argmax(probs))
    label = model.config.id2label[pred_idx]
    confidence = float(probs[pred_idx])

    top3_vals, top3_idx = torch.topk(probs, k=min(3, probs.numel()))

    top3 = [
        {"label": model.config.id2label[int(i)], "score": float(s)}
        for s, i in zip(top3_vals, top3_idx)
    ]

    severity = "mild" if confidence < 0.4 else ("moderate" if confidence < 0.75 else "severe")

    return {
        "category": label,
        "confidence": confidence,
        "severity": severity,
        "top3": top3,
    }



def _has_emotion_keywords(text: str, lang: str) -> bool:
    text = text.lower()
    kws = EMOTION_KEYWORDS.get(lang, EMOTION_KEYWORDS["tr"])
    return any(k in text for k in kws)


def is_relevant_to_mental_state(text: str, lang: str, is_in_session: bool = False) -> bool:
    
    text = (text or "").strip()
    if not text:
        return False

    if is_in_session:
        return True

    if _has_emotion_keywords(text, lang):
        return True

    if normalize_short_answer(text, lang) != "UNKNOWN":
        return True

    
    if len(text.split()) > 1:
        return True

    return False


def redirect_user(lang):
    return random.choice(REDIRECT_REPLIES.get(lang, REDIRECT_REPLIES["tr"]))



# ================== Recommendations DB ==================

COMPREHENSIVE_RECOMMENDATIONS = {
    "Anxiety": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "4-4-4 nefes egzersizini deneyin.",
                    "Kısa bir yürüyüş yaparak sakinleşin.",
                    "Gerilim hissettiğinizde omuzlarınızı gevşetin."
                ],
                "kendi_bakımınız": [
                    "Günlük 10–15 dakika meditasyon yapın.",
                    "Kafein tüketimini azaltın.",
                    "Uyku düzeninizi koruyun."
                ],
                "uzman_destegi": "Kaygı yönetimi için psikolojik danışmanlık alabilirsiniz.",
                "gelecek_adimlar": [
                    "Kaygı tetikleyicilerinizi belirleyin.",
                    "Günlük duygusal durum takibi yapın.",
                    "Nefes egzersizlerini düzenli uygulayın."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Kas gevşetme teknikleri uygulayın.",
                    "Endişelerinizi yazın ve objektif değerlendirin.",
                    "10 dakika sakinleşme molası verin."
                ],
                "kendi_bakımınız": [
                    "Düzenli spor yapın.",
                    "Gün içinde kısa farkındalık molaları ekleyin.",
                    "Duygusal yük oluşturan ortamlardan uzaklaşın."
                ],
                "uzman_destegi": "Bir terapist ile kaygı yönetimi çalışmaları yapabilirsiniz.",
                "gelecek_adimlar": [
                    "Bilişsel davranışçı terapi değerlendirin.",
                    "Daha iyi kriz yönetimi becerileri geliştirin.",
                    "Gerekirse tıbbi destek alın."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Derin nefes alarak bedeninizi sakinleştirin.",
                    "Kendinizi güvende hissettiğiniz bir alana geçin.",
                    "Yakın bir arkadaşınızla iletişim kurun."
                ],
                "kendi_bakımınız": [
                    "Gün içinde kısa sakinleşme araları planlayın.",
                    "Stresli durumlardan uzak durmaya çalışın.",
                    "Uyku düzeninizi iyileştirmeye odaklanın."
                ],
                "uzman_destegi": "Yoğun kaygı belirtileri için psikiyatrik değerlendirme gereklidir.",
                "gelecek_adimlar": [
                    "Düzenli terapiye başlayın.",
                    "Gerekirse ilaç tedavisi değerlendirin.",
                    "Kaygı yönetim planı oluşturun."
                ]
            }
        }
    },
    "Bipolar": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "Duygudurum değişikliklerinizi takip edin.",
                    "Kısa bir yürüyüş yapın.",
                    "Kafein tüketimini sınırlayın."
                ],
                "kendi_bakımınız": [
                    "Uyku düzeninizi koruyun.",
                    "Hafif egzersizler yapın.",
                    "Günlük enerji seviyenizi not edin."
                ],
                "uzman_destegi": "Bir psikiyatristle görüşmek duygudurum düzenlemeye yardımcı olabilir.",
                "gelecek_adimlar": [
                    "Duygudurum günlüğü tutun.",
                    "Tetikleyici durumları tespit edin.",
                    "Basit bir günlük plan oluşturun."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Fazla uyarıcı aktivitelerden uzak durun.",
                    "Güvendiğiniz biriyle konuşun.",
                    "Uyku düzeninizi sabitleyin."
                ],
                "kendi_bakımınız": [
                    "Düşük yoğunlukta sporlar yapın.",
                    "Alkol ve kafeini azaltın.",
                    "Düzenli bir günlük rutin oluşturun."
                ],
                "uzman_destegi": "Psikiyatrik değerlendirme önerilir.",
                "gelecek_adimlar": [
                    "Terapist desteği alın.",
                    "İlaç tedavisini doktorla değerlendirin.",
                    "Duygudurum yönetimi planı hazırlayın."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Güvende olduğunuz bir alana geçin.",
                    "Aileniz veya arkadaşınızla iletişim kurun.",
                    "Uyarıcı maddelerden tamamen uzak durun."
                ],
                "kendi_bakımınız": [
                    "Aktiviteleri minimuma indirin.",
                    "Sakin bir ortam oluşturun.",
                    "Uyku düzenini kesinlikle bozmayın."
                ],
                "uzman_destegi": "Mani veya ağır depresyon belirtileri için acil psikiyatrik değerlendirme gerekir.",
                "gelecek_adimlar": [
                    "İlaç tedavisine başlanabilir.",
                    "Düzenli terapi alınmalıdır.",
                    "Güvenlik planı hazırlanmalıdır."
                ]
            }
        }
    },
    "Depression": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "Bugün kendiniz için küçük bir hedef belirleyin.",
                    "5 dakikalık bir yürüyüş moralinizi yükseltebilir.",
                    "Yakın bir arkadaşınıza mesaj atmayı deneyin."
                ],
                "kendi_bakımınız": [
                    "Uyku düzeninizi koruyun.",
                    "Kısa günlük egzersizler yapın.",
                    "Duygularınızı yazmak için günlük tutun."
                ],
                "uzman_destegi": "Psikolojik danışmanlık süreci yardımcı olabilir.",
                "gelecek_adimlar": [
                    "Günlük küçük başarılar hedefleyin.",
                    "Destek sisteminizi güçlendirin.",
                    "Keyif aldığınız aktiviteleri plana ekleyin."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Güvendiğiniz biriyle duygularınızı paylaşın.",
                    "Düşüncelerinizi yazarak dışsallaştırın.",
                    "Kendinize kısa bir nefes molası verin."
                ],
                "kendi_bakımınız": [
                    "Düzenli yürüyüşler yapın.",
                    "Enerjinizi azaltan durumlardan uzaklaşın.",
                    "Rutinlerinizi sadeleştirin."
                ],
                "uzman_destegi": "Terapistle görüşmek depresyon yönetimine yardımcı olur.",
                "gelecek_adimlar": [
                    "Düzenli terapi programına katılın.",
                    "Gerekirse psikiyatrik değerlendirme alın.",
                    "Duygusal farkındalık becerilerini geliştirin."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Yalnız kalmamaya çalışın.",
                    "Kendinizi güvende hissedeceğiniz bir ortamda bulunun.",
                    "Acil durumda destek hattına başvurun."
                ],
                "kendi_bakımınız": [
                    "Aktiviteleri azaltın ve basit hedeflere yönelin.",
                    "Nefes egzersizleri uygulayın.",
                    "Uyku düzeninizi sabitleyin."
                ],
                "uzman_destegi": "Yoğun depresyon belirtileri için acil psikiyatrik destek gereklidir.",
                "gelecek_adimlar": [
                    "Tıbbi değerlendirme alın.",
                    "Düzenli terapi ve ilaç desteğine başlayın.",
                    "Günlük güvenlik planı oluşturun."
                ]
            }
        }
    },
    "Stress": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "Kısa bir nefes molası verin.",
                    "Omuzlarınızı gevşetin.",
                    "10 dakikalık bir yürüyüş yapın."
                ],
                "kendi_bakımınız": [
                    "Kafein tüketimini azaltın.",
                    "Uyku düzeninizi iyileştirin.",
                    "Gün içinde küçük meditasyonlar yapın."
                ],
                "uzman_destegi": "Stres yönetimi için danışmanlık yardımcı olabilir.",
                "gelecek_adimlar": [
                    "Günlük stres kaynaklarını belirleyin.",
                    "Rutinlerinizi daha verimli planlayın.",
                    "Gevşeme tekniklerini öğrenin."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Görevlerinizi küçük parçalara bölün.",
                    "Mikro molalar ekleyin.",
                    "Zihninizi boşaltmak için 5 dakikalık duraklama yapın."
                ],
                "kendi_bakımınız": [
                    "Düzenli egzersiz yapın.",
                    "Yavaş nefes teknikleri uygulayın.",
                    "Ekran süresini azaltın."
                ],
                "uzman_destegi": "Terapistten destek almak stresle baş etmeyi kolaylaştırabilir.",
                "gelecek_adimlar": [
                    "Bilişsel stres azaltma yöntemlerini öğrenin.",
                    "Haftalık planlar hazırlayın.",
                    "Kişisel sınırlarınızı belirleyin."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Stres kaynağından uzaklaşın.",
                    "Yakın bir arkadaşla konuşun.",
                    "Derin nefes alarak kendinizi sakinleştirin."
                ],
                "kendi_bakımınız": [
                    "Rutinlerinizi sadeleştirin.",
                    "Aşırı yük oluşturan işleri azaltın.",
                    "Dinlenmeye zaman ayırın."
                ],
                "uzman_destegi": "Şiddetli stres durumunda psikiyatrik destek alınmalıdır.",
                "gelecek_adimlar": [
                    "Profesyonel değerlendirme yapın.",
                    "Terapi sürecine başlayın.",
                    "Günlük stres yönetimi planı oluşturun."
                ]
            }
        }
    },
    "Severe_Psychiatric_Disorders": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "Semptomlarınızı not alın ve takip edin.",
                    "Kısa nefes egzersizleri uygulayın.",
                    "Güvendiğiniz biriyle konuşun."
                ],
                "kendi_bakımınız": [
                    "Uyku düzeninizi koruyun.",
                    "Düşük yoğunluklu egzersizler yapın.",
                    "Aşırı yoğun ortamlardan uzak durun."
                ],
                "uzman_destegi": "Psikiyatrik danışmanlık alınabilir.",
                "gelecek_adimlar": [
                    "Düzenli kontrol randevuları planlayın.",
                    "Tetikleyicilerinizi belirleyin.",
                    "Basit bir günlük plan oluşturun."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Semptomlar arttıysa destek alın.",
                    "Güvende hissedeceğiniz bir alana geçin.",
                    "Birkaç dakika derin nefes alın."
                ],
                "kendi_bakımınız": [
                    "Uyarıcı maddelerden uzak durun.",
                    "Sosyal aktiviteleri sınırlayın.",
                    "Dinlendirici aktiviteler yapın."
                ],
                "uzman_destegi": "Psikiyatristle görüşerek tedavi planı oluşturabilirsiniz.",
                "gelecek_adimlar": [
                    "Gerekirse ilaç tedavisi alın.",
                    "Haftalık terapi programı oluşturun.",
                    "Günlük güvenlik planı yapın."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Yalnız kalmayın.",
                    "Yakın birine haber verin.",
                    "Güvenli bir alanda bekleyin."
                ],
                "kendi_bakımınız": [
                    "Tüm yoğun uyaranlardan uzaklaşın.",
                    "Sakin ve karanlık bir ortam oluşturun.",
                    "Sosyal temasları minimuma indirin."
                ],
                "uzman_destegi": "Acil psikiyatrik müdahale gereklidir.",
                "gelecek_adimlar": [
                    "Hastane değerlendirmesi gerekebilir.",
                    "Yoğun tedavi süreci uygulanabilir.",
                    "Acil durum güvenlik planı oluşturun."
                ]
            }
        }
    },
    "Normal": {
        "tr": {
            "hafif": {
                "anlık_öneriler": [
                    "Kendinizi iyi hissetmeniz harika. Bu durumu koruyun.",
                    "Gün içine kısa bir yürüyüş ekleyebilirsiniz."
                ],
                "kendi_bakımınız": [
                    "Uyku düzeninizi sürdürün.",
                    "Sağlıklı beslenmeye devam edin.",
                    "Hoşlandığınız aktiviteleri artırın."
                ],
                "uzman_destegi": "Şu anda profesyonel yardıma ihtiyaç görünmüyor.",
                "gelecek_adimlar": [
                    "Günlük küçük hedefler belirleyin.",
                    "Sağlıklı alışkanlıklarınızı devam ettirin."
                ]
            },
            "orta": {
                "anlık_öneriler": [
                    "Biraz stresli hissediyor olabilirsiniz, kısa bir mola verin.",
                    "Nefes egzersizi yapın."
                ],
                "kendi_bakımınız": [
                    "Kafein tüketimini azaltın.",
                    "Günlük egzersiz yapın."
                ],
                "uzman_destegi": "Gerekirse bir danışmanla görüşebilirsiniz.",
                "gelecek_adimlar": [
                    "Günlük stres takibi yapın.",
                    "Temel kendi bakım rutinlerini uygulayın."
                ]
            },
            "yüksek": {
                "anlık_öneriler": [
                    "Kendinize sakin bir ortam oluşturun.",
                    "Yakın bir arkadaşla konuşun."
                ],
                "kendi_bakımınız": [
                    "Dinlenmeye öncelik verin.",
                    "Uyku düzeninizi sabitleyin."
                ],
                "uzman_destegi": "Durum kötüleşirse profesyonel destek alın.",
                "gelecek_adimlar": [
                    "Kısa süreli gözlem yapın.",
                    "Gerekirse psikolojik destek alın."
                ]
            }
        }
    },
}


def build_recommendations(category, severity, lang):
    severity_map = {"mild": "hafif", "moderate": "orta", "severe": "yüksek"}
    sev_key = severity_map.get(severity, "hafif")

    block = COMPREHENSIVE_RECOMMENDATIONS.get(category, {})
    pack = block.get(lang) or {}

    if not pack:
        return {}

    sev = pack.get(sev_key, {})

    out = {}
    for k, v in sev.items():
        out[k] = v if isinstance(v, list) else [v]

    return out


def supportive_reply(lang, category):
    bank = {
        "tr": {
            "Depression": [
                "Zor bir dönemden geçtiğini anlıyorum. Bugün küçük bir adım yeterli olabilir.",
                "Bugün minnettar olduğun üç şeyi yazmayı dene.",
            ],
            "Anxiety": [
                "4-4-4 nefesini dene: 4 saniye nefes al, 4 saniye tut, 4 saniye ver.",
                "Endişeni yaz ve gerçekten olasılığını düşün.",
            ],
            "Stress": [
                "Görevleri küçük parçalara ayır ve bir tanesini şimdi yap.",
                "10 dakikalık bir mola ver.",
            ],
            "Default": ["Buradayım.", "Aklından geçenleri biraz daha anlatır mısın?"],
        }
    }

    arr = bank.get(lang, bank["tr"])
    arr = arr.get(category, arr["Default"])
    return random.choice(arr)


def process_user_message(user_id: str, text: str):
    session = SESSIONS[user_id]
    lang = session["lang"]
    step = session["step"]

    timeout_state = check_timeout(session, lang)
    if timeout_state["timeout"]:
        return timeout_state["msg"]

    is_in_session = (step > 0)
    
    if not is_relevant_to_mental_state(text, lang, is_in_session=is_in_session):
        return redirect_user(lang)

    short_label = normalize_short_answer(text, lang)
    text_to_store = text
    
    if short_label != "UNKNOWN" and step < len(QUESTIONS[lang]):
        expanded = expand_short_answer_for_question(step, short_label, lang)
        if expanded:
            text_to_store = expanded

    session["answers"].append(text_to_store)

    if step < len(QUESTIONS[lang]):
        q = QUESTIONS[lang][step]
        session["step"] += 1
        return q

    merged = " ".join(session["answers"])
    
    text_en = translate(merged, lang, "en")
    result = predict_en(text_en)  
    
    session["analysis"] = result

    final_result = {
        "kategori": TR_LABELS.get(result["category"], result["category"]),
        "seviye": TR_SEVERITY.get(result["severity"], result["severity"]),
        "olasilik": result["confidence"],
        "ilk_3": [
            {"etiket": TR_LABELS.get(i["label"], i["label"]), "oran": i["score"]}
            for i in result["top3"]
        ]
    }

    rec = build_recommendations(result["category"], result["severity"], lang)
    support = supportive_reply(lang, result["category"])
    
    session["step"] = 0
    session["answers"] = []

    return {
        "sonuc": final_result,
        "oneriler": rec,
        "destek": support
    }

# ================== CV Model ==================
cv_model = None

class_names = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Severe_Psychiatric_Disorders",
    "Stress",
]


def load_cv():
    global cv_model
    try:
        cv_model = load_model(CV_MODEL_PATH)
        print("CV Model Loaded.")
    except Exception as e:
        cv_model = None
        print(f"Error loading CV model: {e}")


load_cv()


def norm_lang(l):
    if not l:
        return "tr"
    l = l.lower()
    if l.startswith("ar"):
        return "ar"
    if l.startswith("tr"):
        return "tr"
    return "en"


# ================== CV Post-processing & Text Fusion ==================

LABEL_GROUPS = {
    "distress": ["Anxiety", "Stress"],
    "mood_disorder": ["Depression", "Bipolar"],
    "severe": ["Severe_Psychiatric_Disorders"],
    "normal": ["Normal"],
}



KEYWORD_DICTIONARY = {
    "tr": {
        "Anxiety": {
            "kaygı": 1.2,
            "kaygılı": 1.0,
            "endişe": 1.1,
            "endişeliyim": 1.1,
            "panik": 1.3,
            "korku": 1.0,
            "gergin": 0.9,
            "iç sıkıntısı": 1.0,

            # كلمات إضافية
            "kaygım arttı": 1.2,
            "sürekli kaygılanıyorum": 1.3,
            "panik halindeyim": 1.4,
            "çok endişeliyim": 1.2,
            "içim rahat değil": 1.1,
            "içimde bir sıkıntı var": 1.0,
            "telaş içindeyim": 1.1,
            "kendimi huzursuz hissediyorum": 1.1,
            "bir şey olacakmış gibi hissediyorum": 1.1,
            "kalbim hızla atıyor": 1.2,
            "ellerim titriyor": 1.1,
            "nefesim daralıyor": 1.3,
            "gergin hissediyorum": 1.0,
            "akılım karmakarışık": 1.0,
            "çok korkuyorum": 1.2,
            "endişem geçmiyor": 1.3,
            "her an kötü bir şey olacak sanıyorum": 1.3,
            "düşüncelerimi durduramıyorum": 1.2,
            "sürekli tetikteyim": 1.1,
            "göğsüm sıkışıyor": 1.3,
            "kaçmak istiyorum": 1.2,
            "rahatlayamıyorum": 1.1
        },

        "Depression": {
            "depresyon": 1.4,
            "depresif": 1.3,
            "depresyondayım": 1.4,
            "üzgün": 1.1,
            "moralim bozuk": 1.2,
            "çökkün": 1.1,
            "hiçbir şey yapmak istemiyorum": 1.5,
            "enerjim yok": 1.2,

            "hayattan zevk almıyorum": 1.4,
            "kendimi boş hissediyorum": 1.4,
            "hiçbir şey beni mutlu etmiyor": 1.4,
            "karanlık bir ruh halindeyim": 1.5,
            "kendimi kötü hissediyorum": 1.2,
            "yorgunum bitkinim": 1.2,
            "hiç gücüm yok": 1.3,
            "hayat anlamsız geliyor": 1.5,
            "her gün daha kötü hissediyorum": 1.4,
            "kendimi değersiz görüyorum": 1.5,
            "kimse beni anlamıyor": 1.2,
            "yalnızım": 1.1,
            "ağlamak istiyorum": 1.3,
            "kendimi kaybolmuş hissediyorum": 1.4,
            "motivasyonum yok": 1.2,
            "uykum bozuk": 1.1,
            "hep üzgünüm": 1.2,
            "içimde bir ağırlık var": 1.3,
            "hiçbir şey umrumda değil": 1.4,
            "kendimle savaşmaktan yoruldum": 1.4,
            "kendimi toparlayamıyorum": 1.3,
            "ruhum çökmüş gibi": 1.5,
            "karamsarım": 1.2
        },

        "Stress": {
            "stres": 1.4,
            "stresliyim": 1.3,
            "gerginlik": 1.0,
            "baskı altındayım": 1.2,
            "yetişemiyorum": 1.1,
            "yoruldum": 1.0,

            "çok yoruldum": 1.1,
            "aşırı stres altındayım": 1.4,
            "her şey üstüme geliyor": 1.3,
            "kafam dolu": 1.1,
            "zaman yetmiyor": 1.1,
            "dinlenmeye ihtiyacım var": 1.0,
            "mental olarak bitkinim": 1.3,
            "sürekli koşturuyorum": 1.0,
            "odaklanamıyorum": 1.1,
            "tükenmiş hissediyorum": 1.4,
            "çok baskı hissediyorum": 1.2,
            "rahatlayamıyorum": 1.1,
            "her şeyden bıktım": 1.2,
            "kafam karışık": 1.0,
            "stresten patlayacağım": 1.4,
            "gerginim": 1.0,
            "çalışmaktan yoruldum": 1.0,
            "çok yoğun bir dönemden geçiyorum": 1.1,
            "sürekli sıkıntıdayım": 1.2,
            "bitkinim": 1.1,
            "üzerimde baskı var": 1.2,
            "işler kontrolden çıkıyor": 1.3
        },

        "Bipolar": {
            "iniş çıkış": 1.2,
            "duygudurum": 1.0,
            "mani": 1.4,
            "aşırı enerjik": 1.1,
            "bir gün çok iyi bir gün çok kötü": 1.5,

            "ruh halim çok hızlı değişiyor": 1.3,
            "bir anda enerji doluyorum sonra çöküyorum": 1.5,
            "kendimi durduramıyorum": 1.2,
            "aşırı hızlandığım zamanlar oluyor": 1.3,
            "sonra tamamen duruyorum": 1.3,
            "kontrolü kaybediyorum": 1.3,
            "içimde patlama varmış gibi": 1.2,
            "düşüncelerim çok hızlı": 1.3,
            "sonra tamamen boş hissediyorum": 1.4,
            "modum sürekli zıplıyor": 1.3,
            "sanki iki farklı insanım": 1.5,
            "enerjim bir anda fırlıyor": 1.2,
            "sonra yere çakılıyorum": 1.4,
            "duygularım tutarsız": 1.2
        },

        "Severe_Psychiatric_Disorders": {
            "halüsinasyon": 1.5,
            "sesler duyuyorum": 1.6,
            "gerçek değil": 1.3,
            "paranoyak": 1.4,
            "kendime zarar vermek": 1.8,
            "intihar": 2.0,

            "beni izliyorlar": 1.6,
            "birileri beni kontrol ediyor": 1.7,
            "gerçeklik kayboluyor": 1.6,
            "kafamın içinde sesler var": 1.7,
            "rüyadaymışım gibi hissediyorum": 1.4,
            "düşüncelerimi kontrol edemiyorum": 1.4,
            "kabuslar görüyorum": 1.3,
            "her şeyden korkuyorum": 1.4,
            "gerçekle hayali karıştırıyorum": 1.6,
            "insanlara güvenemiyorum": 1.4,
            "biri bana zarar verecek sanıyorum": 1.6,
            "zihnim karışıyor": 1.3,
            "kendime hakim olamıyorum": 1.5,
            "yaşamak istemiyorum": 1.9,
            "varlıklar görüyorum": 1.7
        },

        "Normal": {
            "iyiyim": 1.3,
            "kendimi iyi hissediyorum": 1.5,
            "sorun yok": 1.2,
            "normal hissediyorum": 1.3,
            "her şey yolunda": 1.4,

            "mutluyum": 1.3,
            "rahatım": 1.2,
            "pozitif hissediyorum": 1.2,
            "enerjim yerinde": 1.3,
            "dengede hissediyorum": 1.3,
            "günüm güzel geçiyor": 1.3,
            "kendimi huzurlu hissediyorum": 1.4,
            "zihnim sakin": 1.3,
            "kendimi güçlü hissediyorum": 1.4,
            "olumlu düşünüyorum": 1.3,
            "iyi bir ruh halindeyim": 1.4,
            "hayatım düzenli": 1.3,
            "keyfim yerinde": 1.3,
            "stresim yok": 1.2
        }
    }
}



CV_MIN_PROB = 0.05     
CV_TOP_K = 3           
CV_IMAGE_WEIGHT = 0.6  
CV_TEXT_WEIGHT = 0.4    


def postprocess_cv_probs(raw_probs: np.ndarray) -> np.ndarray:
 
    probs = np.array(raw_probs, dtype=np.float32).flatten()
    if probs.size == 0:
        return probs

    s = probs.sum()
    if s <= 0:
        probs = np.full_like(probs, 1.0 / probs.size)
    else:
        probs /= s

    # threshold
    probs[probs < CV_MIN_PROB] = 0.0

    # top-K smoothing
    k = min(CV_TOP_K, probs.size)
    top_idx = np.argpartition(probs, -k)[-k:]
    mask = np.zeros_like(probs, dtype=bool)
    mask[top_idx] = True
    probs[~mask] = 0.0

    # renormalize after threshold + top-k
    s = probs.sum()
    if s <= 0:
        probs = np.full_like(probs, 1.0 / probs.size)
    else:
        probs /= s

    group_scores = {}
    for gname, labels in LABEL_GROUPS.items():
        idxs = [class_names.index(lbl) for lbl in labels if lbl in class_names]
        if not idxs:
            continue
        group_scores[gname] = float(probs[idxs].sum())

    if group_scores:
        best_group = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_group]
        if best_score >= 0.5:  
            idxs = [class_names.index(lbl) for lbl in LABEL_GROUPS[best_group] if lbl in class_names]
            probs[idxs] *= 1.1
            probs = np.clip(probs, 0.0, 1.0)
            s = probs.sum()
            if s > 0:
                probs /= s

    return probs


def score_text_description(text: str, lang: str) -> np.ndarray:

    
    scores = np.zeros(len(class_names), dtype=np.float32)

    text = (text or "").lower().strip()
    if not text:
        return scores

    lang = norm_lang(lang)
    lang_dict = KEYWORD_DICTIONARY.get(lang) or KEYWORD_DICTIONARY.get("en", {})

    for idx, label in enumerate(class_names):
        kw_map = lang_dict.get(label, {})
        for kw, weight in kw_map.items():
            if kw in text:
                scores[idx] += float(weight)

    total = scores.sum()
    if total > 0:
        scores /= total
    return scores


def fuse_cv_and_text(cv_probs: np.ndarray,  text_scores: np.ndarray) -> np.ndarray:
   

    cv_probs = np.asarray(cv_probs, dtype=np.float32).flatten()
    text_scores = np.asarray(text_scores, dtype=np.float32).flatten()

    # ===== Safety =====
    if text_scores.sum() <= 0:
        return cv_probs / cv_probs.sum()

    # ===== Normalize =====
    text_probs = text_scores / text_scores.sum()
    cv_probs = cv_probs / cv_probs.sum()

    # ===== Strong bias to text =====
    TEXT_WEIGHT = 0.85
    IMAGE_WEIGHT = 0.15

    fused = (
        TEXT_WEIGHT * text_probs
        + IMAGE_WEIGHT * cv_probs
    )

    fused = fused / fused.sum()

    # ===== Hard protection: image cannot override text =====
    text_idx = int(np.argmax(text_probs))
    fused_idx = int(np.argmax(fused))

    if fused_idx != text_idx:
        fused[text_idx] += 0.2
        fused = fused / fused.sum()

    return fused





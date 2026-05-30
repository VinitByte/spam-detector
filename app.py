
import pickle, re, os, numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spam_detector_final.pkl')
with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)

clf    = MODEL['clf']
tfidf  = MODEL['tfidf']
scaler = MODEL['scaler']

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    USE_NLTK = True
except: 
    STOP_WORDS = set()
    USE_NLTK = False

SPAM_KEYWORDS = {
    'free','win','winner','prize','claim','cash','urgent',
    'congratulations','selected','click','offer','limited',
    'guarantee','money','earn','income','deal','buy','cheap'
}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b\d[\d\s]{6,}\d\b', ' phone ', text)
    text = re.sub(r'[\$£€]\s*\d+', ' money ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    if USE_NLTK:
        tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

HAM_KEYWORDS = {
    'meeting', 'schedule', 'report', 'team', 'project',
    'conference', 'invitation', 'register', 'contest',
    'competition', 'round', 'division', 'rating', 'programming',
    'university', 'college', 'professor', 'assignment',
    'interview', 'appointment', 'reminder', 'calendar',
    'noreply', 'unsubscribe', 'official', 'notification'
}

def extract_features(text):
    return {
        'text_length':        len(text),
        'word_count':         len(text.split()),
        'uppercase_ratio':    sum(1 for c in text if c.isupper()) / max(len(text),1),
        'digit_ratio':        sum(1 for c in text if c.isdigit()) / max(len(text),1),
        'exclamation_count':  text.count('!'),
        'question_count':     text.count('?'),
        'url_count':          len(re.findall(r'http\S+|www\S+', text, re.I)),
        'has_phone':          int(bool(re.search(r'\b\d{10,}\b', text))),
        'has_money':          int(bool(re.search(r'[\$£€]\s*\d+', text))),
        'spam_keyword_count': sum(1 for kw in SPAM_KEYWORDS if kw in text.lower()),
        'avg_word_length':    np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'sentence_count':     len(re.split(r'[.!?]+', text)),
        # ── 3 new features added to match the trained scaler ──
        'ham_keyword_count':  sum(1 for kw in HAM_KEYWORDS if kw in text.lower()),
        'has_unsubscribe':    int('unsubscribe' in text.lower()),
        'has_greeting':       int(bool(re.search(r'\bhello\b|\bdear\b|\bhi\b', text.lower()))),
    }

def predict(text):
    clean     = preprocess_text(text)
    tfidf_v   = tfidf.transform([clean])
    hc        = np.array([list(extract_features(text).values())])
    hc_scaled = scaler.transform(hc)
    x         = hstack([tfidf_v, hc_scaled])
    pred      = int(clf.predict(x)[0])
    try:    spam_prob = float(clf.predict_proba(x)[0][1])
    except:
        d = clf.decision_function(x)[0]
        spam_prob = float(1/(1+np.exp(-d)))

    # ── Change this line ──────────────────────────────────
    threshold = 0.70          # was 0.5, now needs 70% confidence to call spam
    pred = 1 if spam_prob > threshold else 0
    # ─────────────────────────────────────────────────────

    return pred, round(spam_prob, 4)

# ── Routes ────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')   # ← renders HTML now

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Send {"text": "your message"}'}), 400
    text = str(data['text']).strip()
    pred, spam_prob = predict(text)
    return jsonify({
        'prediction':       'spam' if pred==1 else 'ham',
        'spam_probability': spam_prob,
        'ham_probability':  round(1-spam_prob, 4),
        'confidence':       spam_prob if pred==1 else round(1-spam_prob, 4),
        'label':            '🚨 SPAM' if pred==1 else '✅ HAM'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL.get('model_name')})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

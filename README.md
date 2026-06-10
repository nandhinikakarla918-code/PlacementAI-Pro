# PlacementAI Pro v3.0
### Portfolio-Grade AIML Platform for BTech Placement Readiness

> Built as a full-stack AIML internship portfolio project demonstrating in-browser ML inference, Claude AI integration, and real-world placement readiness analytics.

---

## 🚀 Quick Start

### Option A — Single File (Easiest)
1. Download `PlacementAI_Pro.html`
2. Open it directly in **Chrome** or any modern browser
3. No server, no install, no dependencies needed

### Option B — Multi-File (For development)
```
PlacementAI/
├── index.html       ← Open this in Chrome
├── styles.css       ← All CSS (32KB)
├── app.js           ← All JS + ML models (122KB)
└── README.md
```
Open `index.html` in Chrome. The CDN scripts (Chart.js, jsPDF) load from the internet.

---

## 📋 Features

### Assessment Flow
| Page | Description |
|------|-------------|
| **Dashboard** | KPI grid — ensemble ML probability, mock score, ATS, company fit |
| **Profile** | 14 academic features that feed into ML models |
| **Skills** | 16 skills with ML weights shown per slider |
| **Mock Test** | 40+ question bank with deduplication across attempts |

### AI Intelligence
| Page | Description |
|------|-------------|
| **Resume Analyzer** | Claude AI → ATS Score + Quality Score + improvements |
| **Analytics** | 3-model ensemble breakdown + feature importance + radar |
| **ML Models** | LR vs DT vs RF comparison — accuracy, confusion matrix, F1 |
| **Companies** | 12 companies with ML readiness scores + skill gaps |
| **Interview Prep** | 5 tabs: Technical, HR, DSA, System Design, Company Specific |
| **Roadmap** | 30/60/90-day AI plans + company-targeted + PDF export |

### Portfolio Analyzers
| Page | Description |
|------|-------------|
| **GitHub Analyzer** | Claude AI scores your GitHub portfolio |
| **LinkedIn Score** | Section-by-section analysis + optimization tips |
| **Progress** | Mock history trend chart + skill level bars |

---

## 🧠 ML Architecture

### Three Models (All run in-browser via JavaScript)

#### 1. Logistic Regression (LR)
- 14 features + 3 interaction terms
- Weights calibrated from Kaggle Campus Placement dataset
- `z = bias + Σ(w_i × normalised_feature_i) + interactions → σ(z)`
- **Accuracy: 84.3% | F1: 83.6%**

#### 2. Decision Tree (DT)
- Simplified Gini-impurity splits
- Root: DSA ≥ 65 → CGPA sub-tree → CP/Communication leaves
- **Accuracy: 82.1% | F1: 81.5%**

#### 3. Random Forest (RF) ← Best Model
- 5 trees with different feature subsets
- Final = average of all tree probabilities
- **Accuracy: 86.7% | F1: 86.0%**

#### Ensemble Formula
```
P(placed) = RF×0.45 + LR×0.35 + DT×0.20
Threshold: P ≥ 0.52 → Placed
```

#### Mock Score Blending
```
dsa_eff   = 0.55 × self_assessed + 0.45 × mock_score
quant_eff = 0.50 × self_assessed + 0.50 × mock_score
```

---

## 🗄️ Feature Engineering

| Feature | Normalisation |
|---------|--------------|
| CGPA | `(cgpa − 5.0) / 5.0` → [0, 1] |
| DSA / Skills | `value / 100` → [0, 1] |
| Internships | `min(value, 3) / 3` → [0, 1] |
| Backlogs | `min(value, 3) / 3` (negative weight: −1.45) |
| College Tier | `(tier − 1) / 2` (negative weight: −0.41) |
| dsa × cp | `0.6 × norm_dsa × norm_cp` (interaction term) |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML5 + CSS3 + JavaScript (ES2022) |
| ML Models | Pure JS — LR, DT, RF implemented from scratch |
| Charts | Chart.js 4.4.0 (radar, bar, line, doughnut) |
| PDF Export | jsPDF 2.5.1 |
| AI Features | Claude API (`claude-sonnet-4-20250514`) |
| Fonts | Instrument Serif + DM Sans + JetBrains Mono |
| Storage | localStorage (`pai_v3`) |

---

## 🔒 Security Note

The Claude API key is handled by the Claude.ai platform when running inside the Claude artifact environment. For standalone deployment:

**Recommended architecture:**
```
Browser (index.html)
    ↓ fetch('/api/claude')
Express.js Backend (Node)
    ↓ ANTHROPIC_API_KEY in .env
Anthropic API
```

Never expose API keys in frontend JavaScript for production.

---

## 📊 Question Bank (Mock Test Deduplication)

- **Pool**: 40+ questions (Technical, Aptitude, Behavioral, Domain)
- **Deduplication**: Tracks last 60 questions per student in localStorage
- **Balancing**: 5 Technical + 3 Aptitude + 2 Behavioral + 2 Domain
- **Weak-skill boost**: Questions matching weak skills get 2× selection weight
- **Recycling**: When pool exhausted, least-recently-used questions recycled

---

## 🏢 Companies Supported

| Tier | Companies |
|------|-----------|
| Product · FAANG | Google, Microsoft, Amazon, Adobe, Flipkart |
| Product · Startup | Zomato/Swiggy |
| Consulting | Deloitte, Accenture |
| Service · Mass | TCS, Infosys, Wipro, Capgemini |

---

## 📁 File Structure

```
PlacementAI/
├── PlacementAI_Pro.html   ← Self-contained single-file app (all-in-one)
├── index.html             ← Multi-file entry point
├── styles.css             ← Design system (glassmorphism dark theme)
├── app.js                 ← ML models + AI calls + all page logic
└── README.md              ← This file
```

---

## 🎓 Portfolio Highlights

This project demonstrates:
- ✅ In-browser ML inference (no Python, no server)
- ✅ 3 ML algorithms with metrics + confusion matrices
- ✅ Claude AI API integration with offline fallbacks
- ✅ 13-page SPA with glassmorphism UI
- ✅ PDF generation, question deduplication, localStorage persistence
- ✅ Responsive design (mobile + desktop)

---

*PlacementAI Pro v3.0 — Built by Nandhini | BTech AI/ML | Bengaluru*

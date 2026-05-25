"""
📰 Fake News Detector
======================
Detects fake/misinformation news headlines using NLP.
Models: Passive Aggressive Classifier + Logistic Regression on TF-IDF.
Dataset: Synthetic labeled dataset (400 headlines) — swap with LIAR / FakeNewsNet for prod.
Author: Anmol Pandey (AnmolPandey9119)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# ─── 1. DATASET ─────────────────────────────────────────────────────
# Realistic synthetic dataset — 200 real + 200 fake headlines
# In production: replace with LIAR dataset, ISOT, or FakeNewsNet

REAL_HEADLINES = [
    "Scientists discover new treatment for Alzheimer's disease in clinical trials",
    "Stock markets rise as Federal Reserve holds interest rates steady",
    "NASA confirms water ice deposits near lunar south pole",
    "World leaders agree to new climate accord at UN summit",
    "COVID-19 vaccine effectiveness study published in New England Journal of Medicine",
    "Apple unveils new MacBook Pro with upgraded M-series chip",
    "Global food prices fell for the third consecutive month, UN agency reports",
    "India launches Chandrayaan-3 mission successfully to lunar orbit",
    "Researchers develop biodegradable plastic alternative from seaweed",
    "Supreme Court rules on landmark voting rights case",
    "Tech giants face new antitrust regulations in European Union",
    "Archaeologists uncover ancient Roman ruins beneath London construction site",
    "Electric vehicle sales surpass 10 million globally for first time",
    "WHO declares end to mpox public health emergency",
    "New study links ultra-processed foods to increased dementia risk",
    "Amazon announces 15000 layoffs citing economic uncertainty",
    "SpaceX Starship completes first successful orbital flight test",
    "ISRO successfully tests its cryogenic engine for Gaganyaan mission",
    "Nobel Prize in Physics awarded for work on quantum entanglement",
    "UN climate report warns of irreversible changes by 2035 without action",
    "India GDP growth forecast revised upward to 6.8% by World Bank",
    "Scientists find evidence of ancient ocean on Mars surface",
    "Government announces new digital public infrastructure framework",
    "ChatGPT reaches 100 million users two months after launch",
    "Researchers discover gene linked to longevity in centenarians",
    "Federal court dismisses lawsuit against major social media platform",
    "New satellite images reveal accelerating Himalayan glacier melt",
    "G20 summit concludes with agreement on debt relief for developing nations",
    "Engineers develop solar panel that works at night using infrared radiation",
    "Study finds Mediterranean diet reduces heart disease risk by 30 percent",
    "Bitcoin ETF approved by SEC opens new era for crypto investment",
    "Breakthrough battery technology promises 1000 km EV range",
    "WHO reports global tuberculosis cases rising after decades of decline",
    "Parliament passes historic data protection bill with strong privacy rules",
    "Astronomers detect strongest fast radio burst signal ever recorded",
    "AI model outperforms doctors in early cancer detection study",
    "India becomes fourth country to land on the moon with Chandrayaan-3",
    "Record 1.4 billion people voted in India general elections",
    "CERN scientists observe new subatomic particle for first time",
    "International Space Station to be deorbited by 2030 NASA confirms",
    "SEBI introduces new framework for algo trading by retail investors",
    "Major earthquake strikes eastern Turkey killing dozens",
    "Semiconductor shortage eases as new fabs come online globally",
    "Mount Everest height officially revised using GPS survey data",
    "Climate scientists record hottest global average temperature in history",
    "Doctors perform world's first whole-eye transplant in New York",
    "Reliance Jio launches satellite internet service across rural India",
    "Scientists grow human kidney in pig embryo for first time",
    "OpenAI releases GPT-5 with multimodal reasoning capabilities",
    "Finance minister presents union budget with focus on infrastructure spending",
    # more to reach 100
    "IIT researchers develop low-cost water purification system using AI",
    "Global coral reef bleaching event declared for fourth time in history",
    "Paytm receives RBI approval to resume onboarding new customers",
    "DRDO successfully tests hypersonic missile technology in trial",
    "Rare discovery of 68 million year old dinosaur embryo found in China",
    "Tesla recalls 2 million vehicles over autopilot safety concerns",
    "Google DeepMind AI solves decades old protein folding challenge",
    "India and USA sign semiconductor supply chain agreement",
    "Landmark gene therapy trial restores hearing in deaf children",
    "Meta reports first ever quarterly revenue decline",
    "Antarctic sea ice reaches record low extent scientists warn",
    "WHO approves first malaria vaccine for widespread use in Africa",
    "Israel and Hamas agree to temporary ceasefire brokered by Qatar",
    "RBI raises repo rate by 25 basis points to control inflation",
    "Scientists clone endangered Przewalski horse using frozen DNA",
    "New IIT campus to be established in Jammu and Kashmir",
    "ONGC discovers major natural gas reserves off Andhra Pradesh coast",
    "Zomato acquires Blinkit in all-stock deal worth 4447 crore rupees",
    "Study confirms face masks reduced COVID spread by 53 percent",
    "ISRO to launch 50 satellites in single mission by year end",
    "India overtakes China as worlds most populous country UN says",
    "Bird flu detected in dairy cattle in United States for first time",
    "Inflation in India eases to 4.2 percent as food prices moderate",
    "WHO warns of new mpox strain spreading in Central Africa",
    "Researchers develop AI that detects depression from voice patterns",
    "Boeing Starliner successfully docks with International Space Station",
    "India announces 100 billion dollar clean energy investment plan",
    "Scientists detect signs of potential life on exoplanet K2-18b",
    "NASSCOM forecasts Indian IT sector to reach 350 billion dollars by 2026",
    "Doctors use robot to perform complex brain surgery with precision",
    "Google faces 5 billion dollar antitrust fine from European Commission",
    "Climate summit agrees to triple renewable energy capacity by 2030",
    "India set to become 3rd largest economy by 2030 IMF projects",
    "Researchers crack century old mathematical theorem using AI assistance",
    "India launches first solar observatory mission Aditya L1",
    "Tesla opens gigafactory in India after Modi-Musk meeting",
    "Study links long COVID to measurable changes in brain structure",
    "Amazon Web Services launches data center region in Hyderabad",
    "New antibiotic discovered that kills drug-resistant bacteria",
    "Sensex crosses 75000 mark for first time in stock market history",
    "AI model translates ancient Sumerian tablets with 90 percent accuracy",
    "India wins cricket world cup defeating South Africa in final",
    "Chandrayaan-3 Pragyan rover confirms sulfur presence near lunar south pole",
    "Scientists develop nanomedicine that targets cancer cells specifically",
    "Reserve Bank of India cuts rates to stimulate economic growth",
    "James Webb Telescope captures image of most distant galaxy ever seen",
    "India achieves 100 gigawatt solar energy capacity milestone",
    "Doctors successfully transplant lab-grown kidney into human patient",
    "Microsoft acquires AI startup for 650 million dollars to boost Copilot",
    "World Health Assembly approves pandemic treaty after three years of talks",
    "IIT Bombay ranked among top 150 universities globally in QS rankings",
    "Study shows regular exercise reduces Alzheimers risk by 40 percent",
    "China launches crew to new Tiangong space station",
    "India digital payments hit record 14 billion transactions in single month",
]

FAKE_HEADLINES = [
    "Scientists CONFIRM 5G towers are secretly controlling people's thoughts",
    "BREAKING: Government puts mind control chips in COVID vaccines EXPOSED",
    "This one fruit CURES cancer overnight doctors don't want you to know",
    "BOMBSHELL: Moon landing was filmed in a Hollywood studio declassified docs prove",
    "Doctors SHOCKED as man cures diabetes in 3 days with kitchen spice",
    "URGENT: Drinking bleach kills COVID-19 virus White House advisor claims",
    "PROVEN: Earth is flat NASA has been lying for decades insider reveals",
    "Secret globalist agenda to depopulate Earth by 90 percent by 2030",
    "Microchips found in COVID vaccine vials under microscope EXPOSED",
    "Bill Gates admits vaccines are population control tool in secret recording",
    "MIRACLE: Dead woman comes back to life after faith healer prayer",
    "China secretly owns all American farmland plan to poison food supply",
    "Scientists find MERMAIDS living in Pacific Ocean government covers it up",
    "New law will FORCE all citizens to take vaccine or face prison",
    "Leaked Pentagon files show UFOs land daily at Area 51 base",
    "Eating raw garlic for 7 days REVERSES heart disease completely",
    "CONFIRMED: Deep state pedophile ring runs Hollywood and Washington DC",
    "George Soros FUNDING illegal migrant invasion of Europe and USA",
    "Doctors are killing COVID patients on purpose to inflate death numbers",
    "Oil companies have suppressed 200 mpg car engine for 50 years",
    "Chemtrails from planes are government-authorized mass poisoning PROOF",
    "Wi-Fi signals cause cancer in children new independent study finds",
    "EXPLOSIVE: Fauci manufactured COVID in secret Wuhan lab with gain of function",
    "Ancient pyramid texts reveal aliens helped build Egyptian monuments",
    "New world order plot to replace paper money with digital implant chip",
    "Drinking lemon water with baking soda cures stage 4 cancer in weeks",
    "BREAKING: Obama arrested for treason military executes in secret",
    "Reptilian shapeshifters control world governments leaked video proves",
    "Government fluoride in water supply causes IQ loss and docility",
    "Elon Musk is actually a time traveler from 2070 who came to save Earth",
    "Leaked footage shows NASA faking Mars rover footage in Arizona desert",
    "This breathing technique removes all toxins from body in one hour",
    "US military bases have portals to other dimensions declassified docs show",
    "Italian researcher proves COVID was created by 5G radiation not virus",
    "Secret society controls all elections using quantum voting machines",
    "Hospitals paid thousands to mark any death as COVID to inflate numbers",
    "Magnets stick to COVID vaccine injection sites PROOF of metal content",
    "Vatican library holds proof that Jesus survived crucifixion lived to 80",
    "North Korea has cure for all cancers but USA pays to keep it secret",
    "Ancient civilization lived under Antarctica discovered by secret expedition",
    "Flu shot contains mercury levels 25000 times safe limit for children",
    "Holographic technology used to fake entire moon landing video",
    "GM crops contain gene that makes humans infertile secret WHO report shows",
    "Facebook and Google listening to all private conversations through phones",
    "Soros funded antifa cells planning military coup in 20 US cities",
    "Raw milk cures autism in children pediatricians refuse to admit",
    "Sunscreen causes skin cancer more than sun itself big pharma conspiracy",
    "DNA from COVID vaccines permanently alters human genome forever",
    "Ancient Vedic texts describe nuclear weapons used 12000 years ago",
    "BREAKING: CDC whistleblower exposes vaccine autism link cover-up",
    "Secret underground tunnels connect all major world capitals for elites",
    "WHO planning to declare global emergency to seize all national powers",
    "Children in vaccinated families 300 percent more likely to get autism",
    "Rothschild family controls all central banks in the world exposed",
    "BANNED video shows how to cure cancer with hydrogen peroxide at home",
    "Scientists find immortality gene but Big Pharma paid to suppress it",
    "CIA admits to using weather control weapons against enemy nations",
    "Tap water in 50 US cities contains lithium to keep population sedated",
    "Deep state plans to detonate EMP and blame it on Russia for war",
    "Himalayan pink salt mixed with turmeric cures any autoimmune disease",
    "Government admits chemtrails contain barium strontium to dumb down public",
    "Elon Musk Neuralink chips already installed in 10000 unwitting Americans",
    "Leaked WHO documents plan to ban all meat consumption globally by 2030",
    "Pentagon papers show US government had prior knowledge of 9/11",
    "Military scientist turned whistleblower confirms human cloning program",
    "Big Pharma suppresses 100 year old cancer cure that costs pennies",
    "NASA admits aliens visited Earth in 1947 and gave us all technology",
    "Thousands of children disappear into elite global trafficking network yearly",
    "Secret lab creates hybrid human-pig creatures for organ harvesting exposed",
    "George Soros controls supreme court through secret payments BOMBSHELL",
    "Doctors who speak truth about vaccine dangers are being silenced or killed",
    "UN Agenda 2030 is plan to reduce global population to 500 million",
    "Ancient Mayan calendar predicted exact date of world government takeover",
    "Tesla free energy device suppressed by electrical companies for 100 years",
    "Video shows George Floyd is alive and living in Costa Rica under new name",
    "Magnets on arm after COVID booster prove graphene oxide is magnetic",
    "Elite bunkers built under New Zealand for coming global nuclear war",
    "Leaked documents show elections in 47 countries rigged by same firm",
    "Raw apple cider vinegar dissolves kidney stones in 24 hours guaranteed",
    "Ancient text describes how Atlantis was destroyed by sound weapon",
    "FEMA concentration camps activated across USA for political dissidents",
    "Holographic moon being projected to hide real moon replaced with device",
    "Scientist who proved masks don't work found dead in mysterious accident",
    "Russian scientist lived to 140 on daily dose of bear bile and cold water",
    "Leaked Pentagon list shows 8000 famous people are actually cloned",
    "WHO preparing to make vaccines mandatory for all international travel",
    "All mainstream media anchors are paid CIA assets per declassified files",
    "Secret society decides all wars for profit using blackmail of world leaders",
    "New study proves red meat cures depression better than antidepressants",
    "Archaeologists find Noah's Ark on Mount Ararat carbon dated to 5000 BC",
    "Google secretly records all conversations even when phone is off experts claim",
    "COVID PCR test swabs contain self-replicating nano hydrogel surveillance tech",
    "Pilots secretly agree masks cause hypoxia that causes plane crashes",
    "MIT researcher fired for proving WiFi causes irreversible brain damage",
    "Deep state uses Hollywood movies to predict and announce their future plans",
    "Area 51 engineer reveals we have been reverse engineering alien ships since 1952",
    "Drinking colloidal silver cures all bacterial and viral infections instantly",
    "Scientists suppressed proof that cannabis cures all forms of cancer in 1974",
    "NASA astronaut breaks silence admits space is fake and Earth is enclosed dome",
    "Leaked Pfizer documents show company knew vaccine caused heart damage in 2020",
    "Quantum computing breakthrough allows NSA to read all encrypted messages",
]

# Build DataFrame
real_df = pd.DataFrame({"text": REAL_HEADLINES, "label": 0})  # 0 = REAL
fake_df = pd.DataFrame({"text": FAKE_HEADLINES, "label": 1})  # 1 = FAKE
df = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1, random_state=42)

print("=" * 60)
print("📰  FAKE NEWS DETECTOR")
print("=" * 60)
print(f"\n[DATA] Total headlines : {len(df)}")
print(f"[DATA] Real news       : {(df['label']==0).sum()}")
print(f"[DATA] Fake news       : {(df['label']==1).sum()}")


# ─── 2. PREPROCESSING ───────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["text"].apply(preprocess)


# ─── 3. SPLIT & VECTORIZE ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.20, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)
X_train_v = tfidf.fit_transform(X_train)
X_test_v  = tfidf.transform(X_test)

print(f"\n[SPLIT] Train : {len(X_train)}  |  Test : {len(X_test)}")
print(f"[TF-IDF] Vocabulary size : {len(tfidf.vocabulary_)}")


# ─── 4. TRAIN MODELS ────────────────────────────────────────────────
models = {
    "Passive Aggressive": PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
}

print("\n" + "─" * 60)
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("─" * 60)

best_model, best_f1, best_name = None, 0, ""

for name, model in models.items():
    model.fit(X_train_v, y_train)
    y_pred = model.predict(X_test_v)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"{name:<25} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

    if f1 > best_f1:
        best_f1, best_model, best_name = f1, model, name

print("─" * 60)
print(f"\n🏆  Best Model: {best_name}  (F1 = {best_f1:.4f})\n")


# ─── 5. DETAILED EVALUATION ─────────────────────────────────────────
y_pred_best = best_model.predict(X_test_v)
print("─" * 60)
print(f"DETAILED REPORT — {best_name}")
print("─" * 60)
print(classification_report(y_test, y_pred_best, target_names=["Real", "Fake"]))

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  True Negatives  (Real → Real) : {tn}")
print(f"  False Positives (Real → Fake) : {fp}")
print(f"  False Negatives (Fake → Real) : {fn}  ← dangerous misclassifications")
print(f"  True Positives  (Fake → Fake) : {tp}")


# ─── 6. TOP FAKE-NEWS INDICATOR WORDS ───────────────────────────────
if hasattr(best_model, "coef_"):
    print("\n" + "─" * 60)
    print("TOP WORDS STRONGLY ASSOCIATED WITH FAKE NEWS")
    print("─" * 60)
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
    top_fake_idx = np.argsort(coef)[-15:][::-1]
    top_real_idx = np.argsort(coef)[:15]

    print("  Fake indicators  :", list(feature_names[top_fake_idx[:10]]))
    print("  Real indicators  :", list(feature_names[top_real_idx[:10]]))


# ─── 7. CROSS-VALIDATION ────────────────────────────────────────────
cv_scores = cross_val_score(best_model, tfidf.transform(df["clean"]),
                             df["label"], cv=5, scoring="f1")
print(f"\n5-Fold CV F1 Scores : {[round(s,4) for s in cv_scores]}")
print(f"Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─── 8. LIVE PREDICTION DEMO ────────────────────────────────────────
print("\n" + "─" * 60)
print("LIVE PREDICTION DEMO")
print("─" * 60)

test_headlines = [
    "Scientists develop new malaria vaccine with 80 percent efficacy in Africa trials",
    "BOMBSHELL: 5G towers secretly control human DNA government admits",
    "India GDP grows at 7.8 percent in Q3 beating analyst expectations",
    "Drinking apple cider vinegar for 7 days CURES cancer doctors hate this",
    "NASA confirms water ice in permanently shadowed craters on Moon",
    "Global elite planning microchip implants for entire world population EXPOSED",
]

for headline in test_headlines:
    clean   = preprocess(headline)
    vec     = tfidf.transform([clean])
    pred    = best_model.predict(vec)[0]
    verdict = "🚨 FAKE" if pred == 1 else "✅ REAL"
    print(f"  {verdict}  |  {headline[:70]}{'...' if len(headline)>70 else ''}")

print("\n" + "=" * 60)
print("✅  Fake News Detector complete!")
print("=" * 60)

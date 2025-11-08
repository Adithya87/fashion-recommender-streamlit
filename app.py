import streamlit as st
import traceback
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import lightgbm as lgb
import joblib


# ----------------------------
# ⚙️ Global Error Handler
# ----------------------------
def show_error(e):
    st.error(f"❌ {type(e).__name__}: {e}")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


# ----------------------------
# 🎨 Helper Functions
# ----------------------------
def detect_skin_color(image):
    """Extract average skin color from lower-cheek regions using MediaPipe FaceMesh."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        st.error("❌ No face detected. Please upload a clear frontal image.")
        return None

    h, w, _ = img_rgb.shape
    face = result.multi_face_landmarks[0]
    x_coords = [int(lm.x * w) for lm in face.landmark]
    y_coords = [int(lm.y * h) for lm in face.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    face_img = img_rgb[y_min:y_max, x_min:x_max]
    if face_img.size == 0:
        st.error("❌ Could not crop face properly.")
        return None

    h, w, _ = face_img.shape
    left_cheek = face_img[int(h * 0.45):int(h * 0.65), int(w * 0.05):int(w * 0.2)]
    right_cheek = face_img[int(h * 0.45):int(h * 0.65), int(w * 0.8):int(w * 0.95)]
    cheeks = np.vstack([left_cheek.reshape(-1, 3), right_cheek.reshape(-1, 3)])
    avg_skin_rgb = np.mean(cheeks, axis=0)
    return avg_skin_rgb.astype(int)


def predict_outfit(skin_rgb, models):
    """Predict top-3 shirts and 3 pants for each shirt."""
    shirt_model, pant_model, scaler_s, scaler_p, shirt_centers, pant_centers = models
    df_skin = pd.DataFrame([skin_rgb], columns=["Skin Color_R", "Skin Color_G", "Skin Color_B"])
    s_scaled = scaler_s.transform(df_skin)
    s_probs = shirt_model.predict(s_scaled)[0]
    s_top3_idx = np.argsort(s_probs)[-3:][::-1]
    shirts = shirt_centers[s_top3_idx]

    pant_suggestions = []
    for shirt in shirts:
        df_shirt = pd.DataFrame([shirt], columns=["Clothes Color_R", "Clothes Color_G", "Clothes Color_B"])
        p_scaled = scaler_p.transform(df_shirt)
        p_probs = pant_model.predict_proba(p_scaled)[0]
        p_top3_idx = np.argsort(p_probs)[-3:][::-1]
        pant_suggestions.append(pant_centers[p_top3_idx])

    # Best combos = (shirt_i, top pant_i)
    combos = [(shirts[i], pant_suggestions[i][0]) for i in range(len(shirts))]
    return shirts, pant_suggestions, combos


def display_outfits(skin, shirts, pants, combos):
    """Visualize shirts, pants, and final outfit combos."""
    st.subheader("👕👖 Outfit Recommendations")

    # --- Step 1: show skin tone ---
    st.markdown("### 🧍 Detected Skin Tone")
    st.image(np.ones((80, 80, 3), dtype=np.uint8) * skin.reshape(1, 1, 3))

    # --- Step 2: top 3 shirts and their pants ---
    st.markdown("### 👕 Top 3 Shirts and Matching Pants")
    for i, (shirt, pant_colors) in enumerate(zip(shirts, pants), 1):
        st.markdown(f"#### 👕 Shirt {i}")
        shirt_cols = st.columns(4)
        shirt_cols[0].image(np.ones((80, 80, 3), dtype=np.uint8) * shirt.reshape(1, 1, 3))
        shirt_cols[0].markdown("**Shirt Color**")

        for j, pant in enumerate(pant_colors, 1):
            shirt_cols[j].image(np.ones((80, 80, 3), dtype=np.uint8) * pant.reshape(1, 1, 3))
            shirt_cols[j].markdown(f"**Pant {j}**")
        st.divider()

    # --- Step 3: final 3 complete combos ---
    st.markdown("### 💫 Top 3 Complete Outfit Combos")
    combo_cols = st.columns(3)
    for i, (shirt, pant) in enumerate(combos, 1):
        with combo_cols[i - 1]:
            st.markdown(f"#### ✨ Combo {i}")
            patch = np.hstack([
                np.ones((100, 100, 3), dtype=np.uint8) * shirt.reshape(1, 1, 3),
                np.ones((100, 100, 3), dtype=np.uint8) * pant.reshape(1, 1, 3),
            ])
            st.image(patch, caption="👕 + 👖", use_container_width=True)
    st.divider()


# ----------------------------
# 🎨 Optional Dark Theme
# ----------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: #f5f5f5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
try:
    st.set_page_config(page_title="AI Fashion Stylist", page_icon="🎨", layout="centered")
    st.title("🎨 AI Fashion Stylist — Personalized Outfit Recommender")
    st.write("Upload your face photo to get the top-3 shirts, top-3 pants per shirt, and 3 final combos based on your skin tone.")

    uploaded_file = st.file_uploader("📸 Upload a clear face image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5, format="%.1f")
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")

        if st.button("✨ Generate Outfit Suggestions"):
            with st.spinner("🔄 Loading models..."):
                shirt_model = lgb.Booster(model_file="models/shirt_model.txt")
                pant_model = joblib.load("models/pant_model.pkl")
                scaler_s = joblib.load("models/scaler_shirt.pkl")
                scaler_p = joblib.load("models/scaler_pant.pkl")
                shirt_centers = np.load("models/shirt_centers.npy")
                pant_centers = np.load("models/pant_centers.npy")
                models = (shirt_model, pant_model, scaler_s, scaler_p, shirt_centers, pant_centers)

            with st.spinner("🎨 Analyzing skin tone and generating outfits..."):
                skin_rgb = detect_skin_color(image)
                if skin_rgb is not None:
                    st.success(f"Detected Skin RGB: {skin_rgb}")
                    shirts, pants, combos = predict_outfit(skin_rgb, models)
                    display_outfits(skin_rgb, shirts, pants, combos)
                    # --- BMI calculation and body-shape based advice ---
                    def bmi_and_style_advice(weight, height_cm):
                        if height_cm <= 0:
                            return None, None, ["Please enter a valid height."]
                        h_m = height_cm / 100.0
                        bmi = weight / (h_m * h_m)
                        bmi_val = round(bmi, 1)
                        if bmi < 18.5:
                            category = "Underweight"
                            suggestions = [
                                "Slim-fit shirts suit you",
                                "Try fitted/tailored jackets to add shape",
                                "Avoid bulky layers that overwhelm your frame",
                            ]
                        elif bmi < 25:
                            category = "Normal"
                            suggestions = [
                                "Both slim-fit and regular-fit shirts work well",
                                "Tailored suits and structured pieces will look sharp",
                                "You can experiment with slim and regular trouser cuts",
                            ]
                        elif bmi < 30:
                            category = "Overweight"
                            suggestions = [
                                "Structured, slightly looser shirts suit you (avoid very tight slim-fit)",
                                "Choose mid-weight fabrics with some structure",
                                "Straight or slightly tapered pants are flattering",
                            ]
                        else:
                            category = "Obese"
                            suggestions = [
                                "Relaxed or tailored fits suit you — avoid clingy fabrics",
                                "Use vertical lines and darker tones to create a slimming effect",
                                "Structured blazers and open layers add a flattering silhouette",
                            ]
                        return bmi_val, category, suggestions

                    bmi_val, bmi_category, bmi_suggestions = bmi_and_style_advice(weight_kg, height_cm)
                    st.subheader("📏 Body Metrics & Style Advice")
                    if bmi_val is None:
                        st.warning("Please provide a valid height to compute BMI and style suggestions.")
                    else:
                        st.metric(label="BMI", value=f"{bmi_val} ({bmi_category})")
                        st.markdown("**Recommended style tips based on your body type:**")
                        for tip in bmi_suggestions:
                            st.write(f"- {tip}")

except Exception as e:
    show_error(e)

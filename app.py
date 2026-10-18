"""
Platforma Diagnostyki Biomechanicznej
Moduły: Kinematyka AI (Wideo), Kinovea, Biomechanika Biegu, Historia Garmin.
"""

import io
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

# Bezpieczny import MediaPipe (Streamlit Cloud obsługuje go bez problemu)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Konfiguracja i Zmienne Globalne
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Biomechanika AI", page_icon="🏃", layout="wide")

PORTFOLIO_URL = "przykładowy-link"

RUNNING_COLUMNS = ["Dystans", "Tetno", "GCT"]
GARMIN_NUMERIC_COLS = [
    "Dystans", "Średnie tętno", "Średni czas kontaktu z podłożem", 
    "Średnie odchylenie pionowe", "Średnia kadencja biegu", "Suma kalorii"
]

if MP_AVAILABLE:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    JOINT_MAP = {
        "right_arm": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        "left_arm": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        "right_leg": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        "left_leg": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    }

# --------------------------------------------------------------------------- #
# Funkcje Pomocnicze (Matematyka i CSV)
# --------------------------------------------------------------------------- #
def calculate_angle_2d(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def window_mean(series, window):
    if window <= 1: return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try: return raw.decode(encoding)
        except UnicodeDecodeError: continue
    return ""

def _to_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")

# --------------------------------------------------------------------------- #
# Moduł 1: AI VIDEO (MediaPipe + VBT)
# --------------------------------------------------------------------------- #
def render_ai_video_module():
    st.title("🤖 Analiza Wideo AI (MediaPipe) & VBT")
    st.markdown("Wgraj nagranie MP4 z telefonu. Serwer przetworzy ruch, wyliczy kąty, prędkości i pokaże krytyczną klatkę ruchu.")
    
    if not MP_AVAILABLE:
        st.error("Błąd: Biblioteka MediaPipe nie jest zainstalowana. Sprawdź requirements.txt")
        return

    col1, col2 = st.columns(2)
    with col1:
        chain_name = st.selectbox("Badany staw (wierzchołek kąta):", [
            ("right_arm", "Prawa ręka (Bark - Łokieć - Nadgarstek)"),
            ("left_arm", "Lewa ręka (Bark - Łokieć - Nadgarstek)"),
            ("right_leg", "Prawa noga (Biodro - Kolano - Kostka)"),
            ("left_leg", "Lewa noga (Biodro - Kolano - Kostka)"),
        ], format_func=lambda x: x[1])[0]
    with col2:
        target_angle = st.number_input("Kąt docelowy / referencyjny [°]:", min_value=10, max_value=180, value=90)

    uploaded_video = st.file_uploader("Wgraj plik wideo (MP4, MOV)", type=['mp4', 'mov'])

    if uploaded_video:
        if st.button("🚀 Rozpocznij Analizę AI", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            progress_bar = st.progress(0, text="Uruchamianie silnika AI...")
            
            times, angles = [], []
            min_angle_frame = None
            min_angle_val = 999.0
            
            idx_top, idx_mid, idx_bot = JOINT_MAP[chain_name]

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    if frame_count % max(1, (total_frames//100)) == 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"Przetwarzanie klatki {frame_count}/{total_frames}")

                    h, w = frame.shape[:2]
                    if w > 640:
                        frame = cv2.resize(frame, (640, int(h * (640/w))))

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        p1 = (lm[idx_top].x, lm[idx_top].y)
                        p2 = (lm[idx_mid].x, lm[idx_mid].y)
                        p3 = (lm[idx_bot].x, lm[idx_bot].y)

                        if lm[idx_top].visibility > 0.4 and lm[idx_mid].visibility > 0.4 and lm[idx_bot].visibility > 0.4:
                            angle = calculate_angle_2d(p1, p2, p3)
                            t = frame_count / fps
                            times.append(t)
                            angles.append(angle)

                            if angle < min_angle_val:
                                min_angle_val = angle
                                mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                                min_angle_frame = image_rgb.copy()

            cap.release()
            progress_bar.progress(1.0, text="Analiza zakończona! Generowanie raportu...")

            if len(angles) < 5:
                st.error("Nie udało się zidentyfikować sylwetki na wystarczającej liczbie klatek.")
                return

            df = pd.DataFrame({"Czas": times, "Kat": angles})
            df['Kat'] = window_mean(df['Kat'], window=5)
            
            dt = np.gradient(df['Czas'])
            df['Predkosc'] = np.gradient(df['Kat'], df['Czas'])
            df['Predkosc'] = window_mean(df['Predkosc'], window=3)
            
            df['Przyspieszenie'] = np.gradient(df['Predkosc'], df['Czas'])

            min_a = df['Kat'].min()
            max_vel = df['Predkosc'].abs().max()
            status = "✅ Zaliczony" if min_a <= target_angle else "❌ Zbyt płytko"
            color_status = "normal" if min_a <= target_angle else "inverse"

            st.divider()
            
            col_chart, col_img = st.columns([2, 1])
            
            with col_img:
                st.subheader("Moment krytyczny (Najgłębsze zejście)")
                if min_angle_frame is not None:
                    st.image(min_angle_frame, caption=f"Minimalny kąt: {min_a:.1f}°", use_column_width=True)
                
                st.metric("Ocena wzorca (Cel: ≤"+str(target_angle)+"°)", status, delta_color=color_status)
                st.metric("Pik Prędkości Koncentrycznej (VBT)", f"{max_vel:.0f} °/s")

            with col_chart:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df['Czas'], y=df['Kat'], mode='lines', name='Kąt', line=dict(color='#4f46e5', width=3)))
                fig1.add_hline(y=target_angle, line_dash="dash", line_color="#f97316", annotation_text="Kąt referencyjny")
                fig1.update_layout(title="Trajektoria Kąta Zgięcia", xaxis_title="Czas [s]", yaxis_title="Kąt [°]", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=df['Czas'], y=df['Predkosc'], name="Prędkość [°/s]", line=dict(color="#10b981")), secondary_y=False)
                fig2.add_trace(go.Scatter(x=df['Czas'], y=df['Przyspieszenie'], name="Przyspieszenie [°/s²]", line=dict(color="#f59e0b", dash="dot")), secondary_y=True)
                fig2.update_layout(title="Profil Dynamiki (VBT)", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------- #
# Moduł 2: KINEMATYKA KINOVEA
# --------------------------------------------------------------------------- #
def render_kinovea_module():
    st.title("📐 Kinematyka (Eksport Kinovea)")
    st.markdown("Wgraj plik CSV z programu Kinovea (wymagane markery: `Gora`, `Srodek`, `Dol`).")
    
    target_angle = st.number_input("Kąt docelowy [°]:", value=90)
    uploaded = st.file_uploader("Wgraj CSV z Kinovea", type=["csv"], key="kine")
    
    if uploaded:
        try:
            df = pd.read_csv(io.StringIO(_decode_bytes(uploaded.getvalue())), sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            df = df.apply(_to_numeric).dropna().reset_index(drop=True)
            
            angles = []
            for _, row in df.iterrows():
                p1 = (row.get('Gora_X'), row.get('Gora_Y'))
                p2 = (row.get('Srodek_X'), row.get('Srodek_Y'))
                p3 = (row.get('Dol_X'), row.get('Dol_Y'))
                if all(x is not None and not np.isnan(x) for x in p1+p2+p3):
                    angles.append(calculate_angle_2d(p1, p2, p3))
            
            if not angles:
                st.error("Brak poprawnych markerów (Gora, Srodek, Dol).")
                return
                
            df['Kat'] = angles
            min_a = min(angles)
            st.metric("Najmniejszy Kąt", f"{min_a:.1f}°")
            
            fig = go.Figure(go.Scatter(x=df['Czas'], y=df['Kat'], mode="lines", name="Kąt"))
            fig.add_hline(y=target_angle, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd odczytu: {e}")

# --------------------------------------------------------------------------- #
# Moduł 3: BIOMECHANIKA BIEGU
# --------------------------------------------------------------------------- #
def render_running_module():
    st.title("👟 Biomechanika biegu (Pojedynczy trening)")
    uploaded = st.file_uploader("Wgraj CSV z Garmina (kilometry)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(io.StringIO(_decode_bytes(uploaded.getvalue())), sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            df = df[RUNNING_COLUMNS].apply(_to_numeric).dropna().sort_values("Dystans").reset_index(drop=True)
            
            window = st.slider("Wygładzanie wykresu", 1, 15, 3)
            df['Tetno_sm'] = window_mean(df['Tetno'], window)
            df['GCT_sm'] = window_mean(df['GCT'], window)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df["Dystans"], y=df['Tetno_sm'], name="Tętno", line=dict(color="red")), secondary_y=False)
            fig.add_trace(go.Scatter(x=df["Dystans"], y=df['GCT_sm'], name="GCT", line=dict(color="blue")), secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Błąd: {e}")

# --------------------------------------------------------------------------- #
# Moduł 4: HISTORIA GARMIN
# --------------------------------------------------------------------------- #
def render_garmin_history_module():
    st.title("📈 Historia Garmin")
    uploaded = st.file_uploader("Wgraj plik Activities.csv", type=["csv"], key="garmin")
    if uploaded:
        try:
            df = pd.read_csv(io.StringIO(_decode_bytes(uploaded.getvalue())), sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            
            available_cols = [c for c in GARMIN_NUMERIC_COLS if c in df.columns]
            df[available_cols] = df[available_cols].apply(_to_numeric)
            df = df.dropna(subset=available_cols)
            
            c1, c2 = st.columns(2)
            x_axis = c1.selectbox("Oś X", available_cols, index=0)
            y_axis = c2.selectbox("Oś Y", available_cols, index=min(1, len(available_cols)-1))
            
            df = df.sort_values('Data')
            df['Data_str'] = df['Data'].dt.strftime('%Y-%m-%d')
            
            fig = px.scatter(df, x=x_axis, y=y_axis, size="Dystans", color="Data", hover_name="Data_str")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd: {e}")

# --------------------------------------------------------------------------- #
# Główna Nawigacja
# --------------------------------------------------------------------------- #
PAGES = {
    "🏠 Strona główna": lambda: st.write("## 🏃 Analiza Biomechaniki\nWybierz moduł z paska bocznego. W najnowszej wersji dostępna jest automatyczna analiza MP4!"),
    "🤖 AI Wideo & VBT": render_ai_video_module,
    "📐 Kinematyka CSV (Kinovea)": render_kinovea_module,
    "👟 Biomechanika biegu": render_running_module,
    "📈 Historia Garmin": render_garmin_history_module,
}

def main():
    st.sidebar.title("Nawigacja")
    choice = st.sidebar.radio("Wybierz sekcję:", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("System Diagnostyki Ruchowej")
    PAGES[choice]()

if __name__ == "__main__":
    main()

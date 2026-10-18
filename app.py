"""
Platforma Diagnostyki Biomechanicznej - Wersja Hybrydowa (Edge AI)
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

st.set_page_config(page_title="Biomechanika AI", page_icon="🏃", layout="wide")

GARMIN_NUMERIC_COLS = [
    "Dystans", "Średnie tętno", "Średni czas kontaktu z podłożem", 
    "Średnie odchylenie pionowe", "Średnia kadencja biegu", "Suma kalorii"
]

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
# Moduł 1: AI VIDEO (Działa w przeglądarce - Zero obciążenia serwera)
# --------------------------------------------------------------------------- #
def render_ai_video_module():
    st.title("🤖 Analiza Wideo AI (Edge Computing)")
    st.markdown("Moduł przetwarza nagranie bezpośrednio w procesorze Twojego urządzenia, całkowicie omijając limity serwerów Streamlit.")
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    </head>
    <body class="bg-white p-4 font-sans text-gray-800">
        <div class="bg-gray-50 p-5 rounded-xl border mb-6">
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label class="block font-medium mb-1 text-sm">Badany staw:</label>
                    <select id="ai-chain" class="w-full p-2 border rounded-lg text-sm">
                        <option value="right_arm">Prawa ręka (Bark-Łokieć-Nadgarstek)</option>
                        <option value="left_arm">Lewa ręka (Bark-Łokieć-Nadgarstek)</option>
                        <option value="right_leg">Prawa noga (Biodro-Kolano-Kostka)</option>
                        <option value="left_leg">Lewa noga (Biodro-Kolano-Kostka)</option>
                    </select>
                </div>
                <div>
                    <label class="block font-medium mb-1 text-sm">Kąt celu [°]:</label>
                    <input type="number" id="ai-target" value="90" class="w-full p-2 border rounded-lg text-sm">
                </div>
            </div>
            <input type="file" id="video-file" accept="video/mp4,video/quicktime" class="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-50 file:text-blue-700">
            <button onclick="startAnalysis()" id="btn-start" class="bg-blue-600 text-white px-5 py-2 rounded-lg text-sm font-medium">Rozpocznij Analizę</button>
            <div id="status" class="mt-3 text-sm text-blue-600 hidden">Ładowanie sztucznej inteligencji...</div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <video id="video" playsinline muted class="hidden"></video>
            <canvas id="canvas" class="w-full h-auto bg-black rounded-lg hidden"></canvas>
            <div id="metrics" class="hidden grid grid-cols-2 gap-2 content-start"></div>
        </div>
        
        <div id="chart-angle" class="w-full h-[300px] hidden mb-4"></div>
        <div id="chart-vel" class="w-full h-[300px] hidden"></div>

        <script>
            const JOINT_MAP = { "right_arm": [12,14,16], "left_arm": [11,13,15], "right_leg": [24,26,28], "left_leg": [23,25,27] };
            let times=[], angles=[];
            
            const pose = new Pose({locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`});
            pose.setOptions({ modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
            pose.onResults(onResults);

            function calcAngle(p1, p2, p3) {
                let v1 = {x: p1.x-p2.x, y: p1.y-p2.y}, v2 = {x: p3.x-p2.x, y: p3.y-p2.y};
                let dot = v1.x*v2.x + v1.y*v2.y;
                let mag = Math.sqrt(v1.x*v1.x+v1.y*v1.y) * Math.sqrt(v2.x*v2.x+v2.y*v2.y);
                return mag === 0 ? 0 : Math.acos(Math.max(-1, Math.min(1, dot/mag))) * 180 / Math.PI;
            }

            async function startAnalysis() {
                const file = document.getElementById('video-file').files[0];
                if(!file) return alert("Wgraj plik wideo");
                
                times=[]; angles=[];
                document.getElementById('canvas').classList.remove('hidden');
                document.getElementById('status').classList.remove('hidden');
                document.getElementById('btn-start').disabled = true;

                const vid = document.getElementById('video');
                vid.src = URL.createObjectURL(file);
                vid.onloadeddata = () => {
                    document.getElementById('canvas').width = vid.videoWidth;
                    document.getElementById('canvas').height = vid.videoHeight;
                    vid.play();
                    processFrame();
                };
                vid.onended = () => generateReport();
            }

            async function processFrame() {
                const vid = document.getElementById('video');
                if(vid.paused || vid.ended) return;
                await pose.send({image: vid});
                requestAnimationFrame(processFrame);
            }

            function onResults(res) {
                const ctx = document.getElementById('canvas').getContext('2d');
                ctx.clearRect(0,0, canvas.width, canvas.height);
                ctx.drawImage(res.image, 0, 0);
                
                if(res.poseLandmarks) {
                    drawConnectors(ctx, res.poseLandmarks, POSE_CONNECTIONS, {color: '#ffffff', lineWidth: 2});
                    drawLandmarks(ctx, res.poseLandmarks, {color: '#3b82f6', lineWidth: 1});
                    
                    let chain = JOINT_MAP[document.getElementById('ai-chain').value];
                    let p1 = res.poseLandmarks[chain[0]], p2 = res.poseLandmarks[chain[1]], p3 = res.poseLandmarks[chain[2]];
                    
                    if(p1.visibility > 0.4 && p2.visibility > 0.4 && p3.visibility > 0.4) {
                        angles.push(calcAngle(p1, p2, p3));
                        times.push(document.getElementById('video').currentTime);
                    }
                }
            }

            function generateReport() {
                document.getElementById('status').innerText = "Zakończono!";
                if(angles.length < 5) return alert("Nie wykryto sylwetki na nagraniu.");
                
                let minA = Math.min(...angles);
                let target = parseFloat(document.getElementById('ai-target').value);
                
                let vel = [0];
                for(let i=1; i<angles.length-1; i++) {
                    let dy = angles[i+1] - angles[i-1], dx = times[i+1] - times[i-1];
                    vel.push(dx===0 ? 0 : dy/dx);
                }
                vel.push(0);
                let maxV = Math.max(...vel.map(Math.abs));

                document.getElementById('metrics').classList.remove('hidden');
                document.getElementById('metrics').innerHTML = `
                    <div class="p-3 border rounded"><div class="text-xs text-gray-500">Min. Kąt</div><div class="text-xl font-bold">${minA.toFixed(1)}°</div></div>
                    <div class="p-3 border rounded"><div class="text-xs text-gray-500">Ocena (cel: ${target}°)</div><div class="text-xl font-bold">${minA<=target ? '✅' : '❌'}</div></div>
                    <div class="p-3 border rounded"><div class="text-xs text-gray-500">Pik Prędkości</div><div class="text-xl font-bold text-green-600">${maxV.toFixed(0)} °/s</div></div>
                `;

                document.getElementById('chart-angle').classList.remove('hidden');
                document.getElementById('chart-vel').classList.remove('hidden');
                
                Plotly.newPlot('chart-angle', [{x:times, y:angles, mode:'lines', line:{color:'blue'}}], {title:'Trajektoria Zgięcia', margin:{t:30,b:30,l:30,r:30}});
                Plotly.newPlot('chart-vel', [{x:times, y:vel, mode:'lines', line:{color:'green'}}], {title:'Prędkość (VBT)', margin:{t:30,b:30,l:30,r:30}});
            }
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=900, scrolling=True)

# --------------------------------------------------------------------------- #
# Moduł 2: HISTORIA GARMIN
# --------------------------------------------------------------------------- #
def render_garmin_module():
    st.title("📈 Historia Garmin")
    uploaded = st.file_uploader("Wgraj plik Activities.csv", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(io.StringIO(_decode_bytes(uploaded.getvalue())), sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            
            avail = [c for c in GARMIN_NUMERIC_COLS if c in df.columns]
            df[avail] = df[avail].apply(_to_numeric)
            df = df.dropna(subset=avail)
            
            c1, c2 = st.columns(2)
            x_ax = c1.selectbox("Oś X", avail, index=0)
            y_ax = c2.selectbox("Oś Y", avail, index=min(1, len(avail)-1))
            
            df = df.sort_values('Data')
            df['Data_str'] = df['Data'].dt.strftime('%Y-%m-%d')
            fig = px.scatter(df, x=x_ax, y=y_ax, size="Dystans", color="Data", hover_name="Data_str")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd odczytu: {e}")

# --------------------------------------------------------------------------- #
# Nawigacja
# --------------------------------------------------------------------------- #
PAGES = {
    "🤖 AI Wideo (Edge)": render_ai_video_module,
    "📈 Historia Garmin": render_garmin_module,
}

def main():
    st.sidebar.title("Nawigacja")
    choice = st.sidebar.radio("Wybierz moduł:", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()

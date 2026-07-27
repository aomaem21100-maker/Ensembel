from pathlib import Path
import pickle
import joblib

import pandas as pd
import streamlit as st


BASE_PATH = Path(__file__).parent

MODEL_PATH = BASE_PATH / "rf_model.pkl"
FEATURE_NAMES_PATH = BASE_PATH / "feature_names.pkl"
TARGET_NAMES_PATH = BASE_PATH / "target_names.pkl"

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS ที่สวยงามและทันสมัยขึ้น
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Prompt', sans-serif;
        }
        
        .block-container {
            max-width: 1200px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        /* Hero Section */
        .hero {
            padding: 2.5rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            margin-bottom: 2rem;
            color: white;
        }

        .hero h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .hero p {
            margin: 0;
            opacity: 0.95;
            font-size: 1.1rem;
            font-weight: 300;
        }

        /* Input Card */
        .input-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }

        /* Result Card */
        .result-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 1.5rem;
            backdrop-filter: blur(10px);
        }

        .result-success {
            background: linear-gradient(135deg, rgba(72, 187, 120, 0.2) 0%, rgba(56, 142, 60, 0.15) 100%);
            border: 2px solid rgba(72, 187, 120, 0.4);
        }

        /* Metric Cards */
        .metric-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #667eea;
        }

        /* Progress Bar Styling */
        .progress-container {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 0.3rem;
            margin: 0.5rem 0;
        }

        .progress-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            height: 12px;
            transition: width 0.5s ease;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }

        /* Sidebar */
        .sidebar-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
        }

        /* Info Box */
        .info-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }

        /* Number Input Styling */
        .stNumberInput > label {
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        /* Flower Icons */
        .flower-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """โหลดโมเดลและ metadata"""
    
    missing_files = []
    if not MODEL_PATH.exists():
        missing_files.append("rf_model.pkl")
    if not FEATURE_NAMES_PATH.exists():
        missing_files.append("feature_names.pkl")
    if not TARGET_NAMES_PATH.exists():
        missing_files.append("target_names.pkl")
    
    if missing_files:
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {', '.join(missing_files)} "
            "กรุณาวางไฟล์ทั้งหมดไว้ในโฟลเดอร์เดียวกับ app.py"
        )

    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    with open(TARGET_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)

    metadata = {
        "metrics": {"accuracy": 0.96, "roc_auc": 0.99},
        "feature_columns": feature_columns,
        "class_names": class_names,
    }

    return {"model": model, "metadata": metadata}


try:
    model_bundle = load_model()
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_columns = metadata["feature_columns"]
    class_names = metadata["class_names"]
except Exception as error:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {error}")
    st.stop()

# Hero Section
st.markdown(
    """
    <div class="hero">
        <h1>🌸 Iris Flower Classifier</h1>
        <p>ระบบจำแนกพันธุ์ดอกไอริสอัจฉริยะด้วย Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 ข้อมูลโมเดล")
    
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-bottom: 0.3rem;">อัลกอริทึม</div>
                <div style="font-weight: 600; color: white;">Random Forest</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-bottom: 0.3rem;">ประเภทงาน</div>
                <div style="font-weight: 600; color: white;">Classification</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-bottom: 0.3rem;">จำนวนคลาส</div>
                <div style="font-weight: 600; color: white;">3 พันธุ์</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 📈 ความแม่นยำ")
    st.metric("Accuracy", f"{metadata['metrics']['accuracy']:.0%}")
    st.metric("ROC-AUC", f"{metadata['metrics']['roc_auc']:.2f}")
    
    st.markdown("---")
    st.caption("📚 โมเดลนี้จัดทำเพื่อการศึกษา")

# Tabs
single_tab, batch_tab, process_tab = st.tabs([
    " ทำนายรายดอก",
    " ทำนายจาก CSV",
    "️ เกี่ยวกับโมเดล",
])

# Tab 1: Single Prediction
with single_tab:
    st.markdown("###  กรอกข้อมูลดอกไอริส")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.number_input(
            " ความยาวกลีบเลี้ยง (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.1,
            step=0.1,
            help="ความยาวของกลีบเลี้ยง (sepal)"
        )
        
        sepal_width = st.number_input(
            " ความกว้างกลีบเลี้ยง (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="ความกว้างของกลีบเลี้ยง (sepal)"
        )
    
    with col2:
        petal_length = st.number_input(
            "🌺 ความยาวกลีบดอก (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.4,
            step=0.1,
            help="ความยาวของกลีบดอก (petal)"
        )
        
        petal_width = st.number_input(
            "🌺 ความกว้างกลีบดอก (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            help="ความกว้างของกลีบดอก (petal)"
        )
    
    # Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 ทำนายพันธุ์",
            type="primary",
            use_container_width=True,
        )
    
    if predict_button:
        input_data = pd.DataFrame(
            [{
                "sepal length (cm)": sepal_length,
                "sepal width (cm)": sepal_width,
                "petal length (cm)": petal_length,
                "petal width (cm)": petal_width,
            }]
        )
        
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        prediction_label = class_names[prediction]
        
        # ผลลัพธ์
        st.markdown("---")
        st.markdown("### 🎯 ผลการทำนาย")
        
        # แสดงผลพันธุ์ที่ทำนายได้
        icon_map = {
            "setosa": "",
            "versicolor": "", 
            "virginica": ""
        }
        
        flower_icon = icon_map.get(prediction_label.lower(), "")
        
        st.markdown(
            f"""
            <div class="result-card result-success">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 4rem; margin-bottom: 0.5rem;">{flower_icon}</div>
                    <div style="font-size: 1rem; color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">พันธุ์ที่ทำนายได้</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: white;">{prediction_label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # ความน่าจะเป็น
        st.markdown("### 📊 ความน่าจะเป็นของแต่ละพันธุ์")
        
        for i, class_name in enumerate(class_names):
            prob = float(probabilities[i])
            icon = icon_map.get(class_name.lower(), "🌸")
            
            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div>
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                            <span class="metric-label">{class_name}</span>
                        </div>
                        <div class="metric-value" style="color: {'#48bb78' if i == prediction else '#667eea'};">
                            {prob:.1%}
                        </div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {prob*100}%; background: {'linear-gradient(90deg, #48bb78 0%, #38a169 100%)' if i == prediction else 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)'};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # ข้อมูลเพิ่มเติม
        st.markdown(
            """
            <div class="info-box">
                <strong> รู้หรือไม่:</strong> ดอกไอริสแต่ละพันธุ์มีลักษณะเฉพาะที่แตกต่างกัน 
                สามารถสังเกตได้จากความยาวและความกว้างของกลีบดอกและกลีบเลี้ยง
            </div>
            """,
            unsafe_allow_html=True,
        )

# Tab 2: Batch Prediction
with batch_tab:
    st.markdown("### 📁 ทำนายข้อมูลหลายรายการ")
    
    csv_template = pd.DataFrame(
        [{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }]
    )
    
    st.download_button(
        label=" ดาวน์โหลดไฟล์ CSV ตัวอย่าง",
        data=csv_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="iris_prediction_template.csv",
        mime="text/csv",
    )
    
    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV ของคุณ",
        type=["csv"],
        help="ชื่อคอลัมน์ต้องตรงกับไฟล์ตัวอย่าง"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            required_columns = feature_columns
            missing_columns = [
                column
                for column in required_columns
                if column not in batch_data.columns
            ]
            
            if missing_columns:
                st.error(
                    f"❌ ไฟล์ขาดคอลัมน์: {', '.join(missing_columns)}"
                )
            else:
                model_input = batch_data[required_columns].copy()
                
                batch_predictions = model.predict(model_input)
                batch_probabilities = model.predict_proba(model_input)
                
                result_data = batch_data.copy()
                result_data["prediction"] = batch_predictions
                result_data["prediction_label"] = [
                    class_names[int(value)]
                    for value in batch_predictions
                ]
                
                for i, class_name in enumerate(class_names):
                    result_data[f"prob_{class_name}"] = batch_probabilities[:, i].round(4)
                
                st.success(f"✅ ทำนายสำเร็จ {len(result_data):,} รายการ")
                
                st.dataframe(
                    result_data,
                    use_container_width=True,
                    hide_index=True,
                )
                
                st.download_button(
                    label="📥 ดาวน์โหลดผลการทำนาย",
                    data=result_data.to_csv(index=False).encode("utf-8-sig"),
                    file_name="iris_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
                
                # สรุปผล
                st.markdown("### 📊 สรุปผลการทำนาย")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    count_setosa = len(result_data[result_data["prediction_label"] == "setosa"])
                    st.metric(" Setosa", count_setosa)
                
                with col2:
                    count_versicolor = len(result_data[result_data["prediction_label"] == "versicolor"])
                    st.metric(" Versicolor", count_versicolor)
                
                with col3:
                    count_virginica = len(result_data[result_data["prediction_label"] == "virginica"])
                    st.metric(" Virginica", count_virginica)
                
        except Exception as error:
            st.error(f"❌ ไม่สามารถประมวลผลไฟล์ได้: {error}")

# Tab 3: About
with process_tab:
    st.markdown("### ℹ️ เกี่ยวกับโมเดลและข้อมูล")
    
    st.markdown(
        """
        <div class="input-card">
            <h3> Dataset Iris</h3>
            <p>
                Dataset Iris เป็น dataset คลาสสิกในวงการ Machine Learning 
                ประกอบด้วยข้อมูลดอกไอริส 3 พันธุ์ รวม 150 ตัวอย่าง
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 🌿 Features (ตัวแปรนำเข้า)")
    
    schema = pd.DataFrame(
        [
            ["sepal length (cm)", "🌿", "ความยาวของกลีบเลี้ยง (ซม.)"],
            ["sepal width (cm)", "🌿", "ความกว้างของกลีบเลี้ยง (ซม.)"],
            ["petal length (cm)", "🌺", "ความยาวของกลีบดอก (ซม.)"],
            ["petal width (cm)", "🌺", "ความกว้างของกลีบดอก (ซม.)"],
        ],
        columns=["Feature", "Icon", "คำอธิบาย"],
    )
    
    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("### 🌸 พันธุ์ดอกไอริส")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="metric-container" style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌸</div>
                <h4>Setosa</h4>
                <p style="font-size: 0.9rem; color: #666;">
                    พันธุ์ที่มีกลีบดอกเล็กและสั้นที่สุด 
                    แยกจากพันธุ์อื่นได้ง่าย
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="metric-container" style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌺</div>
                <h4>Versicolor</h4>
                <p style="font-size: 0.9rem; color: #666;">
                    พันธุ์ขนาดกลาง 
                    มีลักษณะระหว่าง Setosa และ Virginica
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="metric-container" style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌼</div>
                <h4>Virginica</h4>
                <p style="font-size: 0.9rem; color: #666;">
                    พันธุ์ที่มีกลีบดอกใหญ่และยาวที่สุด
                    มักมีสีม่วงหรือสีน้ำเงิน
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("---")
    st.markdown("### 🤖 Random Forest Algorithm")
    
    st.markdown(
        """
        <div class="info-box">
            <strong>Random Forest</strong> คืออัลกอริทึม Ensemble Learning 
            ที่ใช้ต้นไม้ตัดสินใจ (Decision Trees) หลายต้นร่วมกันทำนายผล 
            โดยแต่ละต้นจะเรียนรู้จากข้อมูลส่วนย่อยและนำผลมาโหวตกัน
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        **ข้อดีของ Random Forest:**
        -  ความแม่นยำสูง
        - 🛡️ ลดโอกาส Overfitting
        -  จัดการกับข้อมูลที่มีมิติสูงได้ดี
        - 🔍 สามารถวัดความสำคัญของ features ได้
        """
    )
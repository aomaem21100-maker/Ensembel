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

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.6rem 1.7rem;
            border-radius: 20px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: linear-gradient(
                135deg,
                rgba(200, 100, 150, 0.09),
                rgba(255, 255, 255, 0.02)
            );
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            font-size: 2rem;
            margin: 0 0 0.35rem 0;
        }

        .hero p {
            margin: 0;
            opacity: 0.78;
        }

        .soft-card {
            padding: 1.2rem 1.3rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 17px;
            margin-top: 1rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 0.8rem;
        }

        .small-note {
            font-size: 0.9rem;
            opacity: 0.70;
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
        "metrics": {"accuracy": 0.0, "roc_auc": 0.0},
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
    st.error(f"ไม่สามารถโหลดโมเดลได้: {error}")
    st.stop()


st.markdown(
    """
    <div class="hero">
        <h1>🌸 Iris Flower Classifier</h1>
        <p>
            ระบบจำแนกพันธุ์ดอกไอริสด้วย Random Forest
            ใช้ข้อมูลความยาวและความกว้างของกลีบดอกและกลีบเลี้ยง
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("ข้อมูลโมเดล")
    st.write("อัลกอริทึม: **Random Forest**")
    st.write("ประเภทงาน: **Classification**")
    st.write("จำนวนคลาส: **3** (Setosa, Versicolor, Virginica)")

    st.divider()
    st.caption(
        "โมเดลและข้อมูลนี้จัดทำเพื่อเป็นตัวอย่างการเรียนรู้ "
        "ไม่ควรใช้ตัดสินจริงโดยไม่มีการตรวจสอบเพิ่มเติม"
    )


single_tab, batch_tab, process_tab = st.tabs(
    [
        "ทำนายรายดอก",
        "ทำนายจาก CSV",
        "ขั้นตอนของโมเดล",
    ]
)


with single_tab:
    st.subheader("กรอกข้อมูลดอกไอริส")

    with st.form("single_prediction_form"):
        left_column, right_column = st.columns(2)

        with left_column:
            sepal_length = st.number_input(
                "ความยาวกลีบเลี้ยง (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.1,
                step=0.1,
            )

            sepal_width = st.number_input(
                "ความกว้างกลีบเลี้ยง (cm)",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
            )

        with right_column:
            petal_length = st.number_input(
                "ความยาวกลีบดอก (cm)",
                min_value=0.0,
                max_value=10.0,
                value=1.4,
                step=0.1,
            )

            petal_width = st.number_input(
                "ความกว้างกลีบดอก (cm)",
                min_value=0.0,
                max_value=10.0,
                value=0.2,
                step=0.1,
            )

        predict_button = st.form_submit_button(
            "ทำนายพันธุ์",
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

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)

        st.success(f"ผลการทำนาย: **{prediction_label}**")

        st.subheader("ความน่าจะเป็นของแต่ละพันธุ์")
        
        for i, class_name in enumerate(class_names):
            prob = float(probabilities[i])
            st.metric(
                class_name,
                f"{prob:.1%}",
            )
            st.progress(prob)

        st.markdown(
            """
            <p class="small-note">
                ข้อมูลจะถูกส่งเข้าสู่ Random Forest โดยตรง
                เพื่อจำแนกพันธุ์ดอกไอริส
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


with batch_tab:
    st.subheader("ทำนายข้อมูลหลายรายการ")

    csv_template = pd.DataFrame(
        [{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }]
    )

    st.download_button(
        label="ดาวน์โหลดไฟล์ CSV ตัวอย่าง",
        data=csv_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="iris_prediction_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ CSV",
        type=["csv"],
        help="ชื่อคอลัมน์ต้องตรงกับไฟล์ตัวอย่าง",
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
                    "ไฟล์ขาดคอลัมน์ต่อไปนี้: "
                    + ", ".join(missing_columns)
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

                st.success(
                    f"ทำนายสำเร็จทั้งหมด {len(result_data):,} รายการ"
                )

                st.dataframe(
                    result_data,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    label="ดาวน์โหลดผลการทำนาย",
                    data=result_data.to_csv(index=False).encode("utf-8-sig"),
                    file_name="iris_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as error:
            st.error(f"ไม่สามารถประมวลผลไฟล์ได้: {error}")


with process_tab:
    st.subheader("กระบวนการทำงานของระบบ")

    st.markdown(
        """
        **1. Input Features**

        - ความยาวกลีบเลี้ยง (sepal length)
        - ความกว้างกลีบเลี้ยง (sepal width)
        - ความยาวกลีบดอก (petal length)
        - ความกว้างกลีบดอก (petal width)

        **2. Prediction**

        - ส่งข้อมูลเข้าสู่ Random Forest
        - แสดงพันธุ์ที่ทำนายและค่าความน่าจะเป็นของแต่ละพันธุ์
        """
    )

    schema = pd.DataFrame(
        [
            ["sepal length (cm)", "ตัวเลข", "ความยาวกลีบเลี้ยง (ซม.)"],
            ["sepal width (cm)", "ตัวเลข", "ความกว้างกลีบเลี้ยง (ซม.)"],
            ["petal length (cm)", "ตัวเลข", "ความยาวกลีบดอก (ซม.)"],
            ["petal width (cm)", "ตัวเลข", "ความกว้างกลีบดอก (ซม.)"],
        ],
        columns=["ชื่อคอลัมน์", "ชนิดข้อมูล", "คำอธิบาย"],
    )

    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### พันธุ์ดอกไอริส")
    
    for class_name in class_names:
        st.write(f"- **{class_name}**")
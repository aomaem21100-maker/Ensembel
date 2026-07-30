from pathlib import Path
import pickle
import os

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("random_forest_model.pkl")

# ==================== Developer Info ====================
DEVELOPER_NAME = "นายจตุรภัทร สถาปิคานนท์"
DEVELOPER_ID = "024"

# ✅ ใช้ URL จากอินเทอร์เน็ต (ปลอดภัย ไม่ error)
DEVELOPER_IMAGE = "https://ui-avatars.com/api/?name=Jaturapat+Sthapik&size=300&background=388E3C&color=fff&bold=true"

# ถ้าอยากใช้รูปตัวเอง ให้เปลี่ยนเป็น URL รูปออนไลน์ เช่น:
# DEVELOPER_IMAGE = "https://your-image-hosting.com/15.jpg"


st.set_page_config(
    page_title="Random Forest Predictor",
    page_icon="🌱",
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
                rgba(56, 142, 60, 0.09),
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

        /* ===== Developer Card Styles ===== */
        .developer-sidebar-card {
            background: linear-gradient(
                135deg,
                rgba(56, 142, 60, 0.12),
                rgba(255, 255, 255, 0.02)
            );
            border: 1px solid rgba(56, 142, 60, 0.25);
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
        }

        .developer-avatar {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid #388E3C;
            box-shadow: 0 4px 12px rgba(56, 142, 60, 0.3);
            margin-bottom: 0.8rem;
        }

        .developer-name {
            color: #2E7D32;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0.3rem 0;
        }

        .developer-id {
            color: #388E3C;
            font-size: 0.95rem;
            font-weight: 600;
            margin: 0.2rem 0;
        }

        .developer-role {
            color: #7f8c8d;
            font-size: 0.85rem;
            font-style: italic;
            margin: 0.2rem 0;
        }

        .footer-developer {
            background: linear-gradient(
                135deg,
                rgba(56, 142, 60, 0.08),
                rgba(255, 255, 255, 0.02)
            );
            border: 1px solid rgba(128, 128, 128, 0.22);
            padding: 1.8rem;
            border-radius: 18px;
            margin-top: 2rem;
            text-align: center;
        }

        .footer-developer-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1.8rem;
            flex-wrap: wrap;
        }

        .footer-developer-text {
            text-align: left;
        }

        .footer-developer-text h3 {
            color: #2E7D32;
            margin: 0 0 0.4rem 0;
        }

        .footer-developer-text .developer-name {
            font-size: 1.3rem;
            margin: 0.3rem 0;
        }

        .footer-developer-text .developer-id {
            font-size: 1rem;
            margin: 0.3rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {MODEL_PATH.name} "
            "กรุณาวางไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับ app.py"
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model_bundle = load_model()
    pipeline = model_bundle["pipeline"]
    metadata = model_bundle["metadata"]
except Exception as error:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {error}")
    st.stop()


st.markdown(
    """
    <div class="hero">
        <h1> Random Forest Predictor</h1>
        <p>
            เว็บตัวอย่างสำหรับทำนายแนวโน้มการสอบผ่าน
            ด้วย Ensemble Learning และ Data Preprocessing Pipeline
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== Sidebar with Developer Info ====================
with st.sidebar:
    st.header("ข้อมูลโมเดล")
    st.write("อัลกอริทึม: **Random Forest**")
    st.write("ประเภทงาน: **Classification**")

    model_metrics = metadata.get("metrics", {})
    st.metric(
        "Test Accuracy",
        f"{model_metrics.get('accuracy', 0):.2%}",
    )
    st.metric(
        "Test ROC-AUC",
        f"{model_metrics.get('roc_auc', 0):.3f}",
    )

    st.divider()

    # ===== Developer Profile in Sidebar =====
    st.markdown("### 👨‍💻 ผู้พัฒนา")

    # ✅ ใช้ try-except เพื่อป้องกัน error
    try:
        st.image(DEVELOPER_IMAGE, width=130)
    except Exception:
        # ถ้ารูปมีปัญหา ให้ใช้ placeholder แทน
        st.image(
            "https://ui-avatars.com/api/?name=Jaturapat+Sthapik&size=150&background=388E3C&color=fff&bold=true",
            width=130
        )

    st.markdown(
        f"""
        <div class="developer-sidebar-card">
            <p class="developer-name">{DEVELOPER_NAME}</p>
            <p class="developer-id">🆔 รหัสนักศึกษา: {DEVELOPER_ID}</p>
            <p class="developer-role">ML with Python Developer</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "โมเดลและข้อมูลนี้จัดทำเพื่อเป็นตัวอย่างการเรียนรู้ "
        "ไม่ควรใช้ตัดสินนักเรียนจริงโดยไม่มีการตรวจสอบเพิ่มเติม"
    )


single_tab, batch_tab, process_tab = st.tabs(
    [
        "ทำนายรายบุคคล",
        "ทำนายจาก CSV",
        "ขั้นตอนของโมเดล",
    ]
)


with single_tab:
    st.subheader("กรอกข้อมูลสำหรับการทำนาย")

    with st.form("single_prediction_form"):
        left_column, right_column = st.columns(2)

        with left_column:
            study_hours = st.number_input(
                "ชั่วโมงอ่านหนังสือต่อวัน",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
            )

            attendance_percent = st.number_input(
                "เปอร์เซ็นต์การเข้าเรียน",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
            )

            assignment_score = st.number_input(
                "คะแนนงานหรือการบ้าน",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
            )

        with right_column:
            previous_gpa = st.number_input(
                "เกรดเฉลี่ยเดิม",
                min_value=0.0,
                max_value=4.0,
                value=2.75,
                step=0.05,
            )

            internet_access = st.selectbox(
                "มีอินเทอร์เน็ตสำหรับการเรียนหรือไม่",
                options=["Yes", "No"],
                format_func=lambda value: "มี" if value == "Yes" else "ไม่มี",
            )

            tutoring = st.selectbox(
                "เข้าร่วมการสอนเสริมหรือไม่",
                options=["Yes", "No"],
                format_func=lambda value: (
                    "เข้าร่วม" if value == "Yes" else "ไม่เข้าร่วม"
                ),
            )

        predict_button = st.form_submit_button(
            "ทำนายผล",
            type="primary",
            use_container_width=True,
        )

    if predict_button:
        input_data = pd.DataFrame(
            [{
                "study_hours": study_hours,
                "attendance_percent": attendance_percent,
                "assignment_score": assignment_score,
                "previous_gpa": previous_gpa,
                "internet_access": internet_access,
                "tutoring": tutoring,
            }]
        )

        prediction = int(pipeline.predict(input_data)[0])
        probabilities = pipeline.predict_proba(input_data)[0]

        fail_probability = float(probabilities[0])
        pass_probability = float(probabilities[1])
        prediction_label = metadata["class_names"][prediction]

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.success(f"ผลการทำนาย: **{prediction_label}**")
        else:
            st.warning(f"ผลการทำนาย: **{prediction_label}**")

        metric_left, metric_right = st.columns(2)

        metric_left.metric(
            "ความน่าจะเป็นที่จะผ่าน",
            f"{pass_probability:.1%}",
        )

        metric_right.metric(
            "ความน่าจะเป็นที่จะไม่ผ่าน",
            f"{fail_probability:.1%}",
        )

        st.progress(pass_probability)

        st.markdown(
            """
            <p class="small-note">
                ข้อมูลจะผ่านขั้นตอนเติมค่าที่หาย ปรับมาตรฐาน
                และแปลงข้อมูลหมวดหมู่โดยอัตโนมัติก่อนเข้าสู่ Random Forest
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


with batch_tab:
    st.subheader("ทำนายข้อมูลหลายรายการ")

    csv_template = pd.DataFrame(
        [{
            "study_hours": 4.0,
            "attendance_percent": 85.0,
            "assignment_score": 75.0,
            "previous_gpa": 2.75,
            "internet_access": "Yes",
            "tutoring": "No",
        }]
    )

    st.download_button(
        label="ดาวน์โหลดไฟล์ CSV ตัวอย่าง",
        data=csv_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="prediction_template.csv",
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

            required_columns = metadata["feature_columns"]
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

                batch_predictions = pipeline.predict(model_input)
                batch_probabilities = pipeline.predict_proba(model_input)[:, 1]

                result_data = batch_data.copy()
                result_data["prediction"] = batch_predictions
                result_data["prediction_label"] = [
                    metadata["class_names"][int(value)]
                    for value in batch_predictions
                ]
                result_data["pass_probability"] = batch_probabilities.round(4)

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
                    file_name="random_forest_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as error:
            st.error(f"ไม่สามารถประมวลผลไฟล์ได้: {error}")


with process_tab:
    st.subheader("กระบวนการทำงานของระบบ")

    st.markdown(
        """
        **1. Numeric Data Preprocessing**

        - เติมค่าที่หายด้วยค่ามัธยฐาน
        - Transform ด้วย StandardScaler

        **2. Categorical Data Preprocessing**

        - เติมค่าที่หายด้วยค่าที่พบบ่อยที่สุด
        - Transform ด้วย One-Hot Encoding

        **3. Prediction**

        - ส่งข้อมูลที่แปลงแล้วเข้าสู่ Random Forest
        - แสดงคลาสที่ทำนายและค่าความน่าจะเป็น
        """
    )

    schema = pd.DataFrame(
        [
            ["study_hours", "ตัวเลข", "ชั่วโมงอ่านหนังสือต่อวัน"],
            ["attendance_percent", "ตัวเลข", "เปอร์เซ็นต์การเข้าเรียน"],
            ["assignment_score", "ตัวเลข", "คะแนนงานหรือการบ้าน"],
            ["previous_gpa", "ตัวเลข", "เกรดเฉลี่ยเดิม 0–4"],
            ["internet_access", "หมวดหมู่", "Yes หรือ No"],
            ["tutoring", "หมวดหมู่", "Yes หรือ No"],
        ],
        columns=["ชื่อคอลัมน์", "ชนิดข้อมูล", "คำอธิบาย"],
    )

    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True,
    )


# ==================== Footer with Developer Info ====================
st.divider()

st.markdown(
    f"""
    <div class="footer-developer">
        <div class="footer-developer-row">
            <div>
                <img src="{DEVELOPER_IMAGE}"
                     class="developer-avatar"
                     onerror="this.src='image/15.jpg'">
            </div>
            <div class="footer-developer-text">
                <h3>👨‍💻 พัฒนาโดย</h3>
                <p class="developer-name">{DEVELOPER_NAME}</p>
                <p class="developer-id">🆔 รหัสนักศึกษา: {DEVELOPER_ID}</p>
                <p class="developer-role">
                    📚 วิชา: การโปรแกรมสำหรับการเรียนรู้ด้วยเครื่องด้วยภาษาไพทอน
                </p>
            </div>
        </div>
        <hr style="margin: 1.2rem 0; border: none; border-top: 1px solid rgba(128,128,128,0.25);">
        <p style="color: #7f8c8d; margin: 0; font-size: 0.95rem;">
            📅 กรกฎาคม 2026 | ️ Built with <b>Streamlit</b> + <b>scikit-learn</b> + <b>Pandas</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
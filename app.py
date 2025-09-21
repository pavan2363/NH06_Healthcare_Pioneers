import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import re  # Needed for safe patient ID

# --- PDF report function ---
def create_pdf_report(original_img_path, detected_img, report_details, patient_id="N/A"):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # --- Title ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Kidney Stone Detection Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(10, 25, 200, 25)

    # --- Page settings ---
    page_width = 210
    margin = 15
    available_width = page_width - 2 * margin
    spacing = 10  # gap between original & detected image
    half_width = (available_width - spacing) / 2
    max_height = 90  # reduced height for better fit

    # --- Resize detected image to fit PDF ---
    img_width, img_height = detected_img.size
    aspect_ratio = img_width / img_height

    def calc_size(max_w, max_h, ar):
        if (max_w / ar) <= max_h:
            return max_w, max_w / ar
        else:
            return max_h * ar, max_h

    pdf_width, pdf_height = calc_size(half_width, max_height, aspect_ratio)

    image_y = 35

    # --- Save detected image temporarily ---
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_detected_img:
        detected_img.save(temp_detected_img.name)
        detected_img_path = temp_detected_img.name

    # --- Save original image temporarily ---
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_orig_img:
        orig_img = Image.open(original_img_path)
        orig_img.save(temp_orig_img.name)
        original_img_temp_path = temp_orig_img.name

    try:
        # --- Labels ---
        pdf.set_font("Arial", 'B', 12)
        pdf.set_xy(margin, image_y - 8)
        pdf.cell(half_width, 8, "Original Scan", border=0, align='C')
        pdf.set_xy(margin + half_width + spacing, image_y - 8)
        pdf.cell(half_width, 8, "Detection Result", border=0, align='C')

        # --- Images side by side ---
        pdf.image(original_img_temp_path, x=margin, y=image_y, w=pdf_width, h=pdf_height)
        pdf.image(detected_img_path, x=margin + half_width + spacing, y=image_y, w=pdf_width, h=pdf_height)

        # --- Detection Analysis ---
        pdf.set_y(image_y + pdf_height + 15)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Detection Analysis", ln=True)

        pdf.set_font("Arial", '', 12)
        for detail in report_details:
            pdf.cell(0, 8, detail, ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Risk Assessment:", ln=True)

        # --- Decide risk based on actual detection confidence ---
        if report_details:  
            confidences = []
            for detail in report_details:
                match = re.search(r"Confidence = ([0-9.]+)", detail)
                if match:
                    confidences.append(float(match.group(1)))
            max_conf = max(confidences) if confidences else 0.0
        else:
            max_conf = 0.0

        if max_conf > 0.25:
            risk_text = "HIGHER RISK - Detected stones are likely significant and should be medically reviewed."
        else:
            risk_text = "LOWER RISK - Stones detected are less confident but still require medical verification."

        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(page_width - 2 * margin, 7, risk_text)

        # --- Footer ---
        pdf.ln(10)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
        pdf.cell(0, 8, f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 8, "Doctor's Signature: ____________________", ln=True)

        pdf.ln(8)
        pdf.set_font("Arial", 'I', 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, "--- End of Report ---", ln=True, align='C')

        pdf_bytes = pdf.output(dest="S")
        return io.BytesIO(pdf_bytes)

    finally:
        os.remove(detected_img_path)
        if os.path.exists(original_img_temp_path):
            os.remove(original_img_temp_path)


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Kidney Stone Detection", page_icon="🩺")

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

# Load YOLO model
model = load_yolo_model("runs/detect/train2/weights/best.pt")

# Sidebar
st.sidebar.title("🩺 Kidney Stone Detector")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
uploaded_file = st.sidebar.file_uploader("📂 Upload a Kidney Scan Image", type=["jpg", "jpeg", "png"])
patient_id = st.sidebar.text_input("🧾 Enter Patient ID", value="P12345")
st.sidebar.info("⚠️ Results are for research purposes and should be verified by a medical professional.")

# Main title
st.title(" Kidney Stone Detection and Analysis")
st.write("Upload a kidney scan image to detect stones and generate a **PDF report** with analysis.")

pixel_per_mm = 3

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_container_width=True)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_orig_img:
            img.save(temp_orig_img.name)
            original_img_path = temp_orig_img.name

    with st.spinner("🔍 Analyzing image... Please wait."):
        results = model.predict(img_np, conf=confidence_threshold)
        res_plotted_pil = Image.fromarray(results[0].plot()[:, :, ::-1])

    with col2:
        st.subheader("✅ Detection Result")
        st.image(res_plotted_pil, caption="Detected Stones", use_container_width=True)

    st.markdown("---")

    if len(results[0].boxes) > 0:
        st.subheader("🔬 Detection Analysis")
        report_lines = []

        with st.expander("📑 Show Detailed Report", expanded=True):
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                size_px = max(x2 - x1, y2 - y1)
                size_mm = size_px / pixel_per_mm
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                kidney_side = "Left Kidney" if center_x < img_np.shape[1] / 2 else "Right Kidney"
                img_height = img_np.shape[0]
                vertical_loc = "Top" if center_y < img_height / 3 else "Middle" if center_y < 2 * img_height / 3 else "Bottom"

                location = f"{vertical_loc} of {kidney_side}"
                detail_str = f"Stone {i+1}: Size = {size_mm:.1f} mm, Location = {location}, Confidence = {box.conf.item():.2f}"
                st.write(detail_str)
                report_lines.append(detail_str)

        total_stones_str = f"Total Stones Detected: {len(results[0].boxes)}"
        st.success(f"✅ {total_stones_str}")

        st.markdown("---")
        st.subheader("📥 Download Full Report")
        with st.spinner("📝 Generating PDF report..."):
            pdf_buffer = create_pdf_report(original_img_path, res_plotted_pil, report_lines, patient_id)
            safe_patient_id = re.sub(r'[^A-Za-z0-9_-]', '_', patient_id.strip())

            st.download_button(
                label="📄 Download Report as PDF",
                data=pdf_buffer,
                file_name=f"{safe_patient_id}_Kidney_Stone_Report.pdf",
                mime="application/pdf"
            )

        os.remove(original_img_path)

    else:
        st.warning(f"No kidney stones were detected with a confidence threshold of {confidence_threshold}.")
        if 'original_img_path' in locals() and os.path.exists(original_img_path):
            os.remove(original_img_path)
else:
    st.info("⬅️ Please upload an image using the sidebar to begin analysis.")


# --- Simple Chatbot (Q&A) ---
st.markdown("---")
st.subheader("💬 Kidney Stone Assistant")

faq = {
    "🧾 What are the next steps after detecting kidney stones?": 
        "👉 Consult a urologist. Next steps often include: drinking more water, pain management, follow-up imaging. Larger stones may require medication or surgery (like lithotripsy).",

    "📊 What is confidence threshold?": 
        "👉 The confidence threshold is the minimum probability required for the model to consider a detection valid. For example, if set to 0.25, only stones detected with at least 25% confidence are reported.",

    "💧 Can kidney stones go away on their own?": 
        "👉 Yes, small stones (under 5 mm) may pass naturally in urine. Larger stones usually need medical treatment.",

    "🥗 How to prevent kidney stones?": 
        "👉 Drink enough water, reduce salt, limit high-oxalate foods (like spinach, nuts), and maintain a healthy diet."
}

for question, answer in faq.items():
    if st.button(question):
        st.info(answer)

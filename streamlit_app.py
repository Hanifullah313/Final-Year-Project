import streamlit as st
import requests
import base64
import json
import io
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from database import init_db, save_record, get_all_records

init_db()

st.set_page_config(
    page_title="Grounded Quantitative Medical AI - Chest Radiograph System",
    page_icon="🫁",
    layout="wide"
)

REQUEST_HEADERS = {
    "bypass-tunnel-reminder": "true",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "StreamlitMedicalApp/1.0"
}

# --- PDF Generation Function ---
def create_clinical_pdf(patient_data, report_text, orig_img_bytes, overlay_img_bytes, ctr_val, ctr_stat, landmarks):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    h1_style = ParagraphStyle('DocHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=colors.HexColor('#0F172A'), alignment=1)
    sub_style = ParagraphStyle('DocSubHeader', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#64748B'), alignment=1)
    sec_heading = ParagraphStyle('SecHeading', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor('#1E3A8A'))
    body_text = ParagraphStyle('CustomBody', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor('#1F2937'))
    caption_text = ParagraphStyle('ImgCaption', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, leading=9, textColor=colors.HexColor('#475569'), alignment=1)
    disclaimer_text = ParagraphStyle('DisclaimerText', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7, leading=9, textColor=colors.HexColor('#DC2626'), alignment=1)

    story = [
        Paragraph("AI-ASSISTED CHEST RADIOGRAPH DIAGNOSTIC REPORT", h1_style),
        Spacer(1, 2),
        Paragraph("Quantitative Grounded Medical Intelligence System (U-Net &bull; LLaVA-1.5-7B QLoRA)", sub_style),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#1E3A8A'), spaceBefore=0, spaceAfter=6)
    ]

    # Demographics Table
    demo_table_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient_data['name']}", body_text),
            Paragraph(f"<b>Age / Gender:</b> {patient_data['age']} yrs / {patient_data['gender']}", body_text),
            Paragraph(f"<b>Date:</b> {datetime.date.today().strftime('%B %d, %Y')}", body_text)
        ],
        [
            Paragraph(f"<b>Patient ID:</b> {patient_data['id']}", body_text),
            Paragraph(f"<b>Modality:</b> Digital Chest X-Ray (PA)", body_text),
            Paragraph(f"<b>CTR Metric:</b> {ctr_val} ({ctr_stat})", body_text)
        ]
    ]
    demo_table = Table(demo_table_data, colWidths=[180, 180, 180])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.extend([demo_table, Spacer(1, 8), Paragraph("QUANTITATIVE CTR CALIPER & SEGMENTATION OVERLAY", sec_heading), Spacer(1, 4)])

    # Images Table
    img1 = RLImage(io.BytesIO(orig_img_bytes), width=2.8*inch, height=2.1*inch)
    img2 = RLImage(io.BytesIO(overlay_img_bytes), width=2.8*inch, height=2.1*inch)
    img_grid = [
        [img1, img2],
        [Paragraph("Original Chest Radiograph", caption_text), Paragraph(f"CTR Calipers & Thoracic Field (CTR: {ctr_val})", caption_text)]
    ]
    img_table = Table(img_grid, colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('TOPPADDING', (0,1), (-1,1), 2),
        ('BOTTOMPADDING', (0,1), (-1,1), 3),
    ]))
    story.extend([img_table, Spacer(1, 6)])

    # Findings Box
    story.extend([Paragraph("AUTOMATED RADIOLOGICAL FINDINGS & IMPRESSION", sec_heading), Spacer(1, 3)])
    findings_table = Table([[Paragraph(report_text, body_text)]], colWidths=[540])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.extend([findings_table, Spacer(1, 8)])

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=0, spaceAfter=4))
    story.append(Paragraph(
        "<b>Notice:</b> This automated preliminary evaluation is generated by an Artificial Intelligence system. "
        "AI can make mistakes. This output does not replace professional medical judgment and must be verified by a certified healthcare professional.",
        disclaimer_text
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Application Tabs ---
tab_diag, tab_history = st.tabs(["🩺 Quantitative Diagnostic Analysis", "📋 Patient Records & History"])

# Sidebar Settings
st.sidebar.header("System Settings")
API_ENDPOINT = st.sidebar.text_input("Backend API URL", value="https://bright-pandas-show.loca.lt")
st.sidebar.markdown("---")
st.sidebar.subheader("Patient Demographics")
p_name = st.sidebar.text_input("Patient Full Name", value="Jane Doe")
col_a, col_b = st.sidebar.columns(2)
p_age = col_a.number_input("Age", min_value=1, max_value=120, value=45)
p_gender = col_b.selectbox("Gender", ["Female", "Male", "Other"])
p_id = st.sidebar.text_input("Patient ID / MRN", value="CXR-94021")

# ==================== TAB 1: NEW ANALYSIS ====================
with tab_diag:
    st.title("Quantitative Grounded Chest X-Ray Diagnostic System")
    st.caption("Cardiothoracic Ratio (CTR) Calipers & Clinical Report Generation via U-Net + LLaVA-7B QLoRA")

    uploaded_file = st.file_uploader("Upload Chest Radiograph (PNG / JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        orig_bytes = uploaded_file.getvalue()

        # Action Button
        if st.button("Calculate CTR & Generate Grounded Report", type="primary", use_container_width=True):
            with st.spinner("Executing Quantitative Grounded Inference on Cloud GPU..."):
                try:
                    files = {"file": (uploaded_file.name, orig_bytes, uploaded_file.type)}
                    response = requests.post(
                        f"{API_ENDPOINT.rstrip('/')}/analyze",
                        files=files,
                        headers=REQUEST_HEADERS,
                        timeout=90
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.session_state["result"] = res_data
                        st.session_state["orig_bytes"] = orig_bytes
                        st.session_state["chat_history"] = [
                            {"role": "user", "content": f"Analyze this chest radiograph with CTR {res_data.get('ctr_value')} ({res_data.get('ctr_status')})."},
                            {"role": "assistant", "content": res_data.get("generated_report", "")}
                        ]

                        orig_b64 = base64.b64encode(orig_bytes).decode('utf-8')
                        record_id = save_record(
                            patient_id=p_id,
                            patient_name=p_name,
                            age=p_age,
                            gender=p_gender,
                            spatial_grounding=f"CTR: {res_data.get('ctr_value')} ({res_data.get('ctr_status')})",
                            clinical_report=res_data.get("generated_report", "N/A"),
                            orig_b64=orig_b64,
                            overlay_b64=res_data.get("heatmap_overlay_base64", "")
                        )
                        st.toast(f"Archived in database (Record #{record_id})", icon="💾")
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

        # Side-by-side Image Container (Aligned Single-Line Headings)
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("### 🩻 Input Radiograph")
            st.image(uploaded_file, use_container_width=True)

        with col_img2:
            st.markdown("### 📏 Grounding Overlay")
            if "result" in st.session_state and st.session_state["result"].get("heatmap_overlay_base64"):
                overlay_bytes = base64.b64decode(st.session_state["result"]["heatmap_overlay_base64"])
                st.image(Image.open(io.BytesIO(overlay_bytes)), use_container_width=True)
            else:
                overlay_bytes = b""
                st.info("CTR caliper overlay will appear here after calculation.")

        # Display Metrics and Findings Once Available
        if "result" in st.session_state and st.session_state["result"]:
            res = st.session_state["result"]
            ctr_val = res.get("ctr_value", 0.45)
            ctr_stat = res.get("ctr_status", "Normal")
            landmarks = res.get("landmarks", [])

            # Quantitative Metrics Banner
            st.markdown("---")
            st.subheader("📏 Quantitative Radiographic Measurements")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(
                    label="Cardiothoracic Ratio (CTR)",
                    value=f"{ctr_val:.3f}",
                    delta="Normal (≤ 0.50)" if ctr_val <= 0.50 else "Cardiomegaly (> 0.50)",
                    delta_color="normal" if ctr_val <= 0.50 else "inverse"
                )
            with m_col2:
                if ctr_val <= 0.50:
                    st.success(f"**Cardiac Silhouette Status:** {ctr_stat}\n\nNo significant cardiomegaly detected.")
                else:
                    st.error(f"**Cardiac Silhouette Status:** {ctr_stat}\n\nPotential cardiac silhouette enlargement identified.")

            # Clinical Report Section
            st.markdown("---")
            st.subheader("📋 Clinical Diagnostic Report")
            st.info(res.get("generated_report", "Clinical analysis complete."))

            patient_payload = {"name": p_name, "age": p_age, "gender": p_gender, "id": p_id}
            pdf_buffer = create_clinical_pdf(
                patient_data=patient_payload,
                report_text=res.get("generated_report", "Normal"),
                orig_img_bytes=st.session_state.get("orig_bytes", b""),
                overlay_img_bytes=overlay_bytes,
                ctr_val=ctr_val,
                ctr_stat=ctr_stat,
                landmarks=landmarks
            )

            st.download_button(
                label="📄 Download Official Quantitative PDF Report",
                data=pdf_buffer,
                file_name=f"CXR_Quantitative_Report_{p_id}.pdf",
                mime="application/pdf",
                type="primary"
            )

            # ----------------- Interactive Multi-turn VLM Chatbot -----------------
            st.markdown("---")
            st.subheader("💬 Interactive Radiologist Copilot (Follow-up Inquiries)")
            st.caption("Ask specific clinical follow-up questions about this scan.")

            quick_cols = st.columns(3)
            sample_q = None
            if quick_cols[0].button("❤️ Analyze Cardiac Silhouette"):
                sample_q = f"The measured CTR is {ctr_val}. Can you evaluate if there are signs of heart failure or pericardial effusion?"
            if quick_cols[1].button("🔍 Check Lung Bases & CP Angles"):
                sample_q = "Are both costophrenic angles sharp and clear of pleural effusion?"
            if quick_cols[2].button("🫁 Inspect Pulmonary Vascularity"):
                sample_q = "Is there any evidence of pulmonary congestion or vascular cephalization?"

            if "chat_history" in st.session_state:
                for msg in st.session_state["chat_history"][2:]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            user_input = st.chat_input("Ask a clinical question about this scan...")
            query_to_send = sample_q or user_input

            if query_to_send and "orig_bytes" in st.session_state:
                st.session_state["chat_history"].append({"role": "user", "content": query_to_send})
                with st.chat_message("user"):
                    st.write(query_to_send)

                with st.chat_message("assistant"):
                    with st.spinner("AI Radiologist reviewing radiograph..."):
                        try:
                            files = {"file": ("radiograph.png", st.session_state["orig_bytes"], "image/png")}
                            data = {"chat_history": json.dumps(st.session_state["chat_history"])}
                            chat_res = requests.post(
                                f"{API_ENDPOINT.rstrip('/')}/chat",
                                files=files,
                                data=data,
                                headers=REQUEST_HEADERS,
                                timeout=90
                            )
                            
                            if chat_res.status_code == 200:
                                reply = chat_res.json().get("reply", "No response generated.")
                                st.write(reply)
                                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                            else:
                                st.error(f"Chat API Error: {chat_res.text}")
                        except Exception as e:
                            st.error(f"Chat error: {e}")

# ==================== TAB 2: DATABASE HISTORY ====================
with tab_history:
    st.title("Patient Diagnostic History Archive")
    records = get_all_records()
    if not records:
        st.info("No records found in local database.")
    else:
        for rec in records:
            with st.expander(f"📁 MRN: {rec.patient_id} — {rec.patient_name} ({rec.created_at.strftime('%Y-%m-%d %H:%M')})"):
                col_h1, col_h2 = st.columns([1, 2])
                with col_h1:
                    st.markdown(f"**Age / Gender:** {rec.age} / {rec.gender}")
                    st.markdown(f"**Grounding:** `{rec.spatial_grounding}`")
                    st.image(Image.open(io.BytesIO(base64.b64decode(rec.overlay_image_b64))), caption="Segmentation & Caliper Overlay", use_container_width=True)
                with col_h2:
                    st.markdown("**Diagnostic Impression:**")
                    st.info(rec.clinical_report)
                    
                    hist_pdf = create_clinical_pdf(
                        patient_data={"name": rec.patient_name, "age": rec.age, "gender": rec.gender, "id": rec.patient_id},
                        report_text=rec.clinical_report,
                        orig_img_bytes=base64.b64decode(rec.original_image_b64),
                        overlay_img_bytes=base64.b64decode(rec.overlay_image_b64),
                        ctr_val=0.45,
                        ctr_stat="Archived",
                        landmarks=[]
                    )
                    st.download_button(
                        label=f"📄 Download PDF for #{rec.patient_id}",
                        data=hist_pdf,
                        file_name=f"Archived_CXR_{rec.patient_id}.pdf",
                        mime="application/pdf",
                        key=f"hist_btn_{rec.id}"
                    )
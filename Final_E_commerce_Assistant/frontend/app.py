import streamlit as st
import requests
from io import BytesIO

# Change this if your backend runs elsewhere
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Smart E-Commerce Assistant",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Smart E-Commerce Assistant")
st.caption("Upload a product image, check description-risk, simulate reviews, and generate a clean report/PDF.")

# -------------------------------------------------------------------
# Session state to reuse data across tabs
# -------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_filename" not in st.session_state:
    st.session_state.image_filename = None

if "image_mime" not in st.session_state:
    st.session_state.image_mime = None

if "description" not in st.session_state:
    st.session_state.description = ""
if "price" not in st.session_state:
    st.session_state.price = 0.0
if "category" not in st.session_state:
    st.session_state.category = ""

if "reviews" not in st.session_state:
    st.session_state.reviews = None

if "report_text" not in st.session_state:
    st.session_state.report_text = None


tab1, tab2, tab3 = st.tabs(["1️⃣ Analyze Image & Description", "2️⃣ Reviews", "3️⃣ Report & PDF"])


# -------------------------------------------------------------------
# Tab 1: Analyze image + description
# -------------------------------------------------------------------
with tab1:
    st.subheader("Step 1: Upload Image & Description")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    with col_right:
        description = st.text_area(
            "Product description",
            value=st.session_state.description or "",
            placeholder="e.g., Premium slim-fit pink phone case featuring a playful panda illustration...",
            height=120,
        )
        price = st.number_input("Price ($)", min_value=0.0, value=float(st.session_state.price or 19.99), step=0.5)
        category = st.text_input(
            "Category",
            value=st.session_state.category or "Phone Case",
        )

        analyze_clicked = st.button("🔍 Run Analysis")

    if analyze_clicked:
        if uploaded_file is None:
            st.error("Please upload an image first.")
        elif not description.strip():
            st.error("Please enter a product description.")
        else:
            with st.spinner("Contacting backend /analyze ..."):
                try:
                    image_bytes = uploaded_file.getvalue()
                    files = {
                        "file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/jpeg")
                    }
                    data = {
                        "description": description,
                        "price": str(price),
                        "category": category,
                    }

                    resp = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data)
                    resp.raise_for_status()
                    result = resp.json()

                    # Store in session for other tabs
                    st.session_state.analysis = result
                    st.session_state.image_bytes = image_bytes
                    st.session_state.image_filename = uploaded_file.name
                    st.session_state.image_mime = uploaded_file.type or "image/jpeg"
                    st.session_state.description = description
                    st.session_state.price = price
                    st.session_state.category = category

                    st.success("Analysis complete ✅")

                except Exception as e:
                    st.error(f"Error calling backend: {e}")

    # Show analysis if available
    if st.session_state.analysis:
        st.markdown("### 🧠 Image & Description Analysis")

        analysis = st.session_state.analysis
        image_analysis = analysis.get("image_analysis", {})
        description_analysis = analysis.get("description_analysis", {})
        comparison = analysis.get("comparison", {})

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**AI Caption (from image):**")
            st.write(image_analysis.get("caption", ""))

            st.markdown("**Image Details:**")
            st.write(
                {
                    "color": image_analysis.get("color"),
                    "material": image_analysis.get("material"),
                    "product_type": image_analysis.get("product_type"),
                    "style": image_analysis.get("style"),
                    "visible_features": image_analysis.get("visible_features", []),
                }
            )

        with c2:
            st.markdown("**Description Analysis:**")
            st.write(description_analysis)

        with c3:
            st.markdown("**Consistency & Risk:**")
            st.write(comparison)

            risk_score = comparison.get("risk_score", 0)
            st.metric("Risk Score (0 = safe, 100 = risky)", risk_score)
            st.progress(min(max(risk_score / 100.0, 0.0), 1.0))


# -------------------------------------------------------------------
# Tab 2: Reviews
# -------------------------------------------------------------------
with tab2:
    st.subheader("Step 2: Simulated Customer Reviews")

    if not st.session_state.analysis:
        st.info("Run the analysis in Tab 1 first.")
    else:
        analysis = st.session_state.analysis
        image_analysis = analysis.get("image_analysis", {})
        caption = image_analysis.get("caption", "")

        st.markdown("**Using Caption from Image:**")
        st.code(caption, language="text")

        st.markdown("You can optionally edit this caption before generating reviews:")

        edited_caption = st.text_input("Caption for review generation", value=caption or "")

        generate_reviews_clicked = st.button("⭐ Generate Reviews")

        if generate_reviews_clicked:
            with st.spinner("Contacting backend /reviews ..."):
                try:
                    data = {
                        "caption": edited_caption,
                        "description": st.session_state.description,
                        "price": str(st.session_state.price),
                        "category": st.session_state.category,
                    }
                    resp = requests.post(f"{BACKEND_URL}/reviews", data=data)
                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state.reviews = result.get("reviews", [])
                    st.success("Reviews generated ✅")
                except Exception as e:
                    st.error(f"Error calling backend: {e}")

        # Show reviews if available
        if st.session_state.reviews:
            st.markdown("### 📣 Simulated Reviews (by Rating)")
            # Sort by rating desc
            sorted_reviews = sorted(
                st.session_state.reviews,
                key=lambda r: r.get("rating", 0),
                reverse=True,
            )
            for r in sorted_reviews:
                rating = r.get("rating", "N/A")
                title = r.get("title", "")
                body = r.get("body", "")
                st.markdown(f"**{rating}-Star Review – {title}**")
                st.write(body)
                st.markdown("---")


# -------------------------------------------------------------------
# Tab 3: Report & PDF
# -------------------------------------------------------------------
with tab3:
    st.subheader("Step 3: Generate Report & Download PDF")

    if not st.session_state.analysis or not st.session_state.image_bytes:
        st.info("Run the analysis in Tab 1 first so we can reuse the image and description.")
    else:
        col_txt, col_pdf = st.columns([2, 1])

        with col_txt:
            generate_report_clicked = st.button("📄 Generate Text Report")

            if generate_report_clicked:
                with st.spinner("Contacting backend /report ..."):
                    try:
                        files = {
                            "file": (
                                st.session_state.image_filename,
                                st.session_state.image_bytes,
                                st.session_state.image_mime,
                            )
                        }
                        data = {
                            "description": st.session_state.description,
                            "price": str(st.session_state.price),
                            "category": st.session_state.category,
                        }
                        resp = requests.post(f"{BACKEND_URL}/report", files=files, data=data)
                        resp.raise_for_status()
                        result = resp.json()
                        st.session_state.report_text = result.get("report", "")
                        st.session_state.reviews = result.get("reviews", [])
                        st.success("Report generated ✅")
                    except Exception as e:
                        st.error(f"Error calling backend: {e}")

            if st.session_state.report_text:
                st.markdown("### 📄 Product Report (Text)")
                st.text(st.session_state.report_text)

        with col_pdf:
            st.markdown("### Export")

            if st.session_state.report_text:
                if st.button("📥 Download PDF Report"):
                    with st.spinner("Contacting backend /report_pdf ..."):
                        try:
                            files = {
                                "file": (
                                    st.session_state.image_filename,
                                    st.session_state.image_bytes,
                                    st.session_state.image_mime,
                                )
                            }
                            data = {
                                "description": st.session_state.description,
                                "price": str(st.session_state.price),
                                "category": st.session_state.category,
                            }
                            resp = requests.post(f"{BACKEND_URL}/report_pdf", files=files, data=data)
                            resp.raise_for_status()
                            pdf_bytes = resp.content

                            st.download_button(
                                label="⬇️ Save Product Report PDF",
                                data=pdf_bytes,
                                file_name="product_report.pdf",
                                mime="application/pdf",
                            )
                        except Exception as e:
                            st.error(f"Error calling backend: {e}")
            else:
                st.caption("Generate the text report first, then you can download the PDF.")

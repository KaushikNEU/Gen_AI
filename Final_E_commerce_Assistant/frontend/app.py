# frontend/app.py

import streamlit as st
import requests
import json
import time
import os

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Smart E-Commerce Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 1rem 2rem;}
    h1 {color: #1e3a8a; margin-bottom: 0.5rem;}
    h2 {color: #1e40af; margin-top: 2rem; margin-bottom: 1rem;}
    h3 {color: #3b82f6; margin-top: 1rem;}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {font-size: 1.8rem !important; font-weight: bold;}
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem;
    }
    
    /* Caption cards */
    .caption-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .caption-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59,130,246,0.15);
    }
    
    .caption-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.75rem;
    }
    
    .caption-text {
        font-size: 1rem;
        color: #374151;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .caption-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-standard {background: #dbeafe; color: #1e40af;}
    .badge-enhanced {background: #d1fae5; color: #065f46;}
    .badge-seo {background: #fef3c7; color: #92400e;}
    
    /* Risk box */
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    
    .risk-low {background: #d1fae5; border-color: #10b981;}
    .risk-medium {background: #fef3c7; border-color: #f59e0b;}
    .risk-high {background: #fee2e2; border-color: #ef4444;}
    
    /* Suggestion cards */
    .suggestion {
        background: white;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    
    .sug-critical {border-color: #dc2626; background: #fef2f2;}
    .sug-high {border-color: #f97316; background: #fff7ed;}
    .sug-medium {border-color: #eab308; background: #fefce8;}
    .sug-low {border-color: #3b82f6; background: #eff6ff;}
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in {
    "analysis": None,
    "image_bytes": None,
    "image_filename": None,
    "image_mime": None,
    "description": "",
    "price": 19.99,
    "category": "Furniture",
    "reviews": None,
    "report_text": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Header
st.title("🛒 Smart E-Commerce Assistant")
st.caption("AI-powered product listing analyzer with intelligent recommendations")

# Quick stats (only if analysis exists)
if st.session_state.analysis:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    comparison = st.session_state.analysis.get("comparison", {})
    risk_score = comparison.get("risk_score", 0)
    similarity = comparison.get("similarity", 0)
    
    with col1:
        st.metric("Risk Score", f"{risk_score}/100", delta=f"{risk_score-50:+d}", delta_color="inverse")
    with col2:
        st.metric("Match Quality", f"{int(similarity*100)}%")
    with col3:
        st.metric("Reviews", len(st.session_state.reviews) if st.session_state.reviews else 0)
    with col4:
        if st.session_state.reviews:
            avg = sum(r.get("rating", 0) for r in st.session_state.reviews) / len(st.session_state.reviews)
            st.metric("Avg Rating", f"{avg:.1f}★")
        else:
            st.metric("Avg Rating", "—")

st.markdown("---")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 Product Analysis", "⭐ Customer Reviews", "📊 Full Report"])

# ============================================================================
# TAB 1: PRODUCT ANALYSIS
# ============================================================================
with tab1:
    # Two-column layout: INPUT (left) | OUTPUT (right)
    left_col, right_col = st.columns([1, 1.4], gap="large")
    
    # ========== LEFT COLUMN: INPUT ==========
    with left_col:
        st.markdown("## 📥 Input")
        
        uploaded_file = st.file_uploader(
            "**Product Image**",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload clear product photo"
        )
        
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        
        st.markdown("---")
        
        description = st.text_area(
            "**Product Description**",
            value=st.session_state.description,
            height=150,
            placeholder="Enter your product description...",
            help="Your current listing description"
        )
        
        col_p, col_c = st.columns(2)
        with col_p:
            price = st.number_input(
                "**Price ($)**",
                min_value=0.0,
                value=float(st.session_state.price),
                step=1.0
            )
        with col_c:
            category = st.selectbox(
                "**Category**",
                ["Phone Case", "Furniture", "Clothing", "Electronics", "Home Decor", "Toys", "Sports", "Other"],
                index=1
            )
        
        st.markdown("")
        analyze_btn = st.button("🔍 **Analyze Product**", type="primary", use_container_width=True)
        
        # Validation & Analysis
        if analyze_btn:
            errors = []
            if not uploaded_file:
                errors.append("📷 Upload an image")
            if not description.strip() or len(description.strip()) < 10:
                errors.append("📝 Enter description (min 10 chars)")
            if price <= 0:
                errors.append("💰 Enter valid price")
            
            if errors:
                for e in errors:
                    st.error(e)
            else:
                with st.spinner("Analyzing..."):
                    try:
                        image_bytes = uploaded_file.getvalue()
                        files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/jpeg")}
                        data = {"description": description, "price": str(price), "category": category}
                        
                        resp = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=60)
                        resp.raise_for_status()
                        
                        st.session_state.analysis = resp.json()
                        st.session_state.image_bytes = image_bytes
                        st.session_state.image_filename = uploaded_file.name
                        st.session_state.image_mime = uploaded_file.type or "image/jpeg"
                        st.session_state.description = description
                        st.session_state.price = price
                        st.session_state.category = category
                        
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
    
    # ========== RIGHT COLUMN: OUTPUT ==========
    with right_col:
        st.markdown("## 📤 AI Analysis Results")
        
        if not st.session_state.analysis:
            st.info("👈 **Upload a product image and click Analyze to see AI-generated insights**")
        else:
            analysis = st.session_state.analysis
            captions = analysis.get("captions", {})
            comparison = analysis.get("comparison", {})
            suggestions = analysis.get("suggestions", [])
            
            # === CAPTIONS ===
            st.markdown("### 🎨 AI-Generated Captions")
            st.caption("Three caption styles optimized for different purposes")
            
            # Standard Caption
            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-title">
                    <span class="caption-badge badge-standard">STANDARD</span>
                    Professional & Factual
                </div>
                <div class="caption-text">{captions.get('standard', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_copy1, col_copy2 = st.columns([3, 1])
            with col_copy2:
                if st.button("📋 Copy", key="copy1", use_container_width=True):
                    st.code(captions.get('standard', ''))
            
            # Enhanced Caption
            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-title">
                    <span class="caption-badge badge-enhanced">ENHANCED</span>
                    With Quality Indicators
                </div>
                <div class="caption-text">{captions.get('enhanced', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_copy3, col_copy4 = st.columns([3, 1])
            with col_copy4:
                if st.button("📋 Copy", key="copy2", use_container_width=True):
                    st.code(captions.get('enhanced', ''))
            
            # SEO Caption
            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-title">
                    <span class="caption-badge badge-seo">SEO OPTIMIZED</span>
                    Keyword-Rich for Search
                </div>
                <div class="caption-text">{captions.get('seo_optimized', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_copy5, col_copy6 = st.columns([3, 1])
            with col_copy6:
                if st.button("📋 Copy", key="copy3", use_container_width=True):
                    st.code(captions.get('seo_optimized', ''))
            
            st.markdown("---")
            
            # === RISK ASSESSMENT ===
            st.markdown("### 🎯 Risk Assessment")
            
            risk_score = comparison.get("risk_score", 0)
            similarity = comparison.get("similarity", 0)
            
            if risk_score < 30:
                risk_class = "risk-low"
                risk_emoji = "🟢"
                risk_label = "LOW RISK"
                risk_msg = "✅ **Excellent!** Image and description align well."
            elif risk_score < 60:
                risk_class = "risk-medium"
                risk_emoji = "🟡"
                risk_label = "MEDIUM RISK"
                risk_msg = "⚠️ **Some issues detected.** Review recommended."
            else:
                risk_class = "risk-high"
                risk_emoji = "🔴"
                risk_label = "HIGH RISK"
                risk_msg = "❌ **Major mismatches found.** Update required."
            
            st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h3 style="margin: 0 0 0.5rem 0; color: #1f2937;">{risk_emoji} {risk_label}</h3>
                <p style="margin: 0; font-size: 1.1rem; color: #374151;">{risk_msg}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                    Risk Score: <strong>{risk_score}/100</strong> | 
                    Similarity: <strong>{int(similarity*100)}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Issues
            if risk_score > 30:
                missing = comparison.get("missing_features", [])
                if similarity < 0.6:
                    st.warning("⚠️ Low similarity between image and description")
                if missing:
                    st.warning(f"⚠️ Missing information: {', '.join(missing)}")
            
            st.markdown("---")
            
            # === AI SUGGESTIONS ===
            if suggestions:
                st.markdown("### 💡 AI Recommendations")
                st.caption("Actionable tips to improve your listing")
                
                for sug in suggestions:
                    priority = sug.get("priority", "low")
                    icon = sug.get("icon", "💡")
                    title = sug.get("title", "")
                    desc = sug.get("description", "")
                    
                    priority_class = f"sug-{priority}"
                    priority_badge = priority.upper()
                    
                    st.markdown(f"""
                    <div class="suggestion {priority_class}">
                        <strong>{icon} {title}</strong> 
                        <span style="font-size: 0.75rem; color: #6b7280;">[{priority_badge}]</span>
                        <p style="margin: 0.5rem 0 0 0; color: #4b5563;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ **No issues found!** Your listing looks great.")
            
            st.markdown("---")
            
            # === IMPROVED DESCRIPTION ===
            with st.expander("✨ Generate AI-Improved Description"):
                if st.button("🤖 Generate", key="gen_improved", use_container_width=True):
                    with st.spinner("Generating..."):
                        try:
                            data = {
                                "description": st.session_state.description,
                                "image_analysis": json.dumps(analysis["image_analysis"]),
                                "price": str(st.session_state.price),
                                "category": st.session_state.category,
                            }
                            resp = requests.post(f"{BACKEND_URL}/improve_description", data=data, timeout=30)
                            resp.raise_for_status()
                            result = resp.json()
                            
                            st.markdown("**🔵 Original:**")
                            st.info(result['original'])
                            
                            st.markdown("**🟢 AI-Improved:**")
                            st.success(result['improved'])
                            
                            if st.button("📋 Copy Improved", key="copy_improved"):
                                st.code(result['improved'])
                        except Exception as e:
                            st.error(f"Error: {e}")

# ============================================================================
# TAB 2: REVIEWS
# ============================================================================
with tab2:
    st.markdown("## ⭐ Customer Review Simulation")
    
    if not st.session_state.analysis:
        st.info("👈 Complete product analysis first")
    else:
        captions = st.session_state.analysis.get("captions", {})
        caption = captions.get("standard", "")
        
        st.markdown(f"**Using caption:** _{caption}_")
        
        if st.button("🎲 Generate Reviews", type="primary", use_container_width=True):
            with st.spinner("Generating realistic reviews..."):
                try:
                    data = {
                        "caption": caption,
                        "description": st.session_state.description,
                        "price": str(st.session_state.price),
                        "category": st.session_state.category,
                    }
                    resp = requests.post(f"{BACKEND_URL}/reviews", data=data, timeout=30)
                    resp.raise_for_status()
                    st.session_state.reviews = resp.json().get("reviews", [])
                    st.success("✅ Reviews generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.reviews:
            st.markdown("---")
            
            for idx, review in enumerate(sorted(st.session_state.reviews, key=lambda r: r.get("rating", 0), reverse=True)):
                rating = review.get("rating", 0)
                title = review.get("title", "")
                body = review.get("body", "")
                stars = "⭐" * rating + "☆" * (5 - rating)
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{stars}** {title}")
                    with col2:
                        st.caption(f"{rating}/5 ⭐")
                    
                    st.markdown(f"_{body}_")
                    st.caption("👤 Verified Purchase • 📅 2 days ago")
                    st.markdown("---")

# ============================================================================
# TAB 3: REPORT
# ============================================================================
with tab3:
    st.markdown("## 📊 Comprehensive Report")
    
    if not st.session_state.analysis:
        st.info("👈 Complete product analysis first")
    else:
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Creating report..."):
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
                    resp = requests.post(f"{BACKEND_URL}/report", files=files, data=data, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state.report_text = result.get("report", "")
                    st.session_state.reviews = result.get("reviews", [])
                    st.success("✅ Report ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.report_text:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                with st.expander("📄 View Text Report", expanded=True):
                    st.text(st.session_state.report_text)
            
            with col2:
                if st.button("📥 Download PDF", use_container_width=True):
                    with st.spinner("Creating PDF..."):
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
                            resp = requests.post(f"{BACKEND_URL}/report_pdf", files=files, data=data, timeout=60)
                            resp.raise_for_status()
                            
                            st.download_button(
                                "⬇️ Save PDF",
                                data=resp.content,
                                file_name="product_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("🤖 Powered by GPT-4 Vision • Smart E-Commerce Assistant v2.0")

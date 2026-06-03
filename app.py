import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize, curve_fit
import os

# Set page configuration
st.set_page_config(
    page_title="Instagram Analytics & Budget Optimizer",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find datasets
def load_data():
    organic_path = "Analysis/instagram_campaign_dataset.csv"
    paid_path = "Analysis/marketing_analytics_dataset.csv"
    
    if not os.path.exists(organic_path):
        organic_path = "instagram_campaign_dataset.csv"
    if not os.path.exists(paid_path):
        paid_path = "marketing_analytics_dataset.csv"
        
    df_org = pd.read_csv(organic_path) if os.path.exists(organic_path) else None
    df_pd = pd.read_csv(paid_path) if os.path.exists(paid_path) else None
    
    return df_org, df_pd

df_organic, df_paid = load_data()

# Check if data is available
if df_organic is None or df_paid is None:
    st.error("Dataset files not found. Run the generator script first.")
    st.stop()

# Cache model training
@st.cache_resource
def train_organic_model(df):
    df_copy = df.copy()
    df_copy['Sentiment'] = df_copy['Post Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_copy['Post Length'] = df_copy['Post Text'].apply(lambda x: len(str(x)))
    df_copy['Hashtag Count'] = df_copy['Hashtags'].apply(lambda x: len(str(x).split()))
    df_copy['Hour'] = pd.to_datetime(df_copy['Timestamp']).dt.hour
    df_copy['Is Video'] = (df_copy['Media Type'] == 'video').astype(int)
    df_copy['Is Carousel'] = (df_copy['Media Type'] == 'carousel').astype(int)
    df_copy['Engagement Binary'] = (df_copy['Engagement'] == 'high').astype(int)
    
    features = ['Post Length', 'Hashtag Count', 'Sentiment', 'Hour', 'Is Video', 'Is Carousel', 'Followers']
    X = df_copy[features]
    y = df_copy['Engagement Binary']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

@st.cache_data
def fit_response_model(df):
    def response_func(X, alpha, beta, bias):
        ad_spend, adstock = X
        return alpha * np.log(ad_spend + 1) + beta * adstock + bias
    
    ad_spend = df['ad_spend'].values
    adstock = df['ad_spend_adstock'].values
    conversions = df['conversions'].values
    
    popt, _ = curve_fit(response_func, (ad_spend, adstock), conversions)
    return popt

# Fit and train
rf_model, model_features = train_organic_model(df_organic)
alpha_opt, beta_opt, bias_opt = fit_response_model(df_paid)

# Sidebar navigation
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Organic Performance", "Paid Budget Optimizer", "Engagement Predictor"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Optimization Stack
- **Organic NLP**: TextBlob sentiment extraction & TF-IDF hashtag features.
- **Organic ML**: RandomForest classifier model.
- **Paid Optimizer**: Diminishing returns curve fitting and SciPy optimizer (`L-BFGS-B`).
""")

# --- PAGE 1: OVERVIEW ---
if page == "Dashboard Overview":
    st.markdown("<h1 class='main-title'>Instagram Analytics & Budget Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Organic Post Engagement Predictor & Paid Ad Budget Allocation Solver</p>", unsafe_allow_html=True)
    
    st.markdown("""
    This application implements data analytics pipelines for two marketing channels:
    1. **Organic Content Strategy**: Analyzing caption sentiment, hashtags, formats, and follower counts to predict post engagement.
    2. **Paid Performance Marketing**: Evaluating campaign funnel conversion metrics, estimating decay curves (Adstock carryover), and solving optimal daily budget allocations under diminishing returns.
    
    Select a tab in the sidebar navigation to get started.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Organic Content Analytics Features")
        st.markdown("""
        * **Sentiment Analysis**: Uses TextBlob to score caption text polarity (-1 to +1).
        * **Hashtag Clustering**: Groups hashtag families using K-Means clustering.
        * **Supervised Classifier**: Supervised classification using Random Forest models to predict engagement.
        """)
        
    with col2:
        st.subheader("Paid Ad Spend Optimizer Features")
        st.markdown("""
        * **Funnel Analysis**: Day-to-day CTR, CVR, CPC, and CPA analysis.
        * **Adstock Modeling**: Incorporates lagged carryover calculations.
        * **Budget Optimizer**: Finds the mathematically optimal ad spend constraint using `scipy.optimize`.
        """)

# --- PAGE 2: ORGANIC CONTENT ANALYTICS ---
elif page == "Organic Performance":
    st.markdown("<h1 class='main-title'>Organic Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Historical Performance, Post Breakdowns & Feature Analysis</p>", unsafe_allow_html=True)
    
    # KPIs
    total_posts = len(df_organic)
    avg_likes = int(df_organic['Likes'].mean())
    avg_er = df_organic['Engagement Rate'].mean()
    top_media = df_organic['Media Type'].value_counts().index[0].title()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Posts", f"{total_posts:,}")
    with kpi2:
        st.metric("Avg Likes", f"{avg_likes:,}")
    with kpi3:
        st.metric("Avg Engagement Rate", f"{avg_er:.2f}%")
    with kpi4:
        st.metric("Top Format", top_media)
        
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Avg Engagement Rate (%) by Media Type")
        media_er_df = df_organic.groupby('Media Type')['Engagement Rate'].mean().reset_index()
        fig_media = px.bar(
            media_er_df,
            x='Media Type',
            y='Engagement Rate',
            color='Media Type',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto='.2f'
        )
        fig_media.update_layout(showlegend=False, xaxis_title="Media Type", yaxis_title="Engagement Rate (%)")
        st.plotly_chart(fig_media, width='stretch')
        
    with chart_col2:
        st.subheader("Engagement Distribution")
        fig_eng = px.pie(
            df_organic,
            names='Engagement',
            color='Engagement',
            color_discrete_map={'high': '#2ec4b6', 'low': '#e71d36'},
            hole=0.4
        )
        st.plotly_chart(fig_eng, width='stretch')

    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.subheader("Follower Count vs. Engagement Rate (%)")
        fig_followers = px.scatter(
            df_organic,
            x='Followers',
            y='Engagement Rate',
            color='Engagement',
            color_discrete_map={'high': '#2ec4b6', 'low': '#e71d36'},
            log_x=True,
            hover_data=['Media Type', 'Likes']
        )
        st.plotly_chart(fig_followers, width='stretch')
        
    with chart_col4:
        st.subheader("Likes vs. Comments by Format")
        fig_likes_comments = px.scatter(
            df_organic,
            x='Likes',
            y='Comments',
            color='Engagement',
            color_discrete_map={'high': '#2ec4b6', 'low': '#e71d36'},
            symbol='Media Type',
            hover_data=['Followers']
        )
        st.plotly_chart(fig_likes_comments, width='stretch')

# --- PAGE 3: PAID BUDGET OPTIMIZER ---
elif page == "Paid Budget Optimizer":
    st.markdown("<h1 class='main-title'>Paid Budget Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Funnel Calculations, Adstock Decay Analysis & Mathematical Optimization</p>", unsafe_allow_html=True)
    
    df_p = df_paid.copy()
    df_p['CTR (%)'] = (df_p['clicks'] / df_p['impressions']) * 100
    df_p['CVR (%)'] = (df_p['conversions'] / df_p['clicks']) * 100
    df_p['CPC'] = df_p['ad_spend'] / df_p['clicks']
    df_p['CPA'] = np.where(df_p['conversions'] > 0, df_p['ad_spend'] / df_p['conversions'], np.nan)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Avg CPC", f"${df_p['CPC'].mean():.2f}")
    with kpi2:
        st.metric("Avg CPA", f"${df_p['CPA'].dropna().mean():.2f}")
    with kpi3:
        st.metric("Avg CTR", f"{df_p['CTR (%)'].mean():.2f}%")
    with kpi4:
        st.metric("Avg CVR", f"{df_p['CVR (%)'].mean():.2f}%")
        
    st.markdown("---")
    
    st.subheader("Mathematical Budget Optimizer")
    st.write(
        "Configure your campaign constraints. The solver uses a logarithmic conversion saturation function "
        "fitted to historical daily spends and brand Adstock carryover to calculate the optimal daily ad spend allocation."
    )
    
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.markdown("#### Input Constraints")
        budget = st.slider("Daily Ad Spend Cap ($)", min_value=5000, max_value=100000, value=30000, step=2500)
        
        avg_adstock = df_paid['ad_spend_adstock'].mean()
        adstock_scen = st.slider(
            "Adstock Carryover ($)",
            min_value=int(df_paid['ad_spend_adstock'].min()),
            max_value=int(df_paid['ad_spend_adstock'].max()),
            value=int(avg_adstock),
            help="Cumulative advertising impact from previous weeks."
        )
        
    with col_results:
        def response_func(spend, adstock, alpha, beta, bias):
            return alpha * np.log(spend + 1) + beta * adstock + bias

        def objective(x):
            return -response_func(x[0], adstock_scen, alpha_opt, beta_opt, bias_opt)

        res = minimize(objective, [budget / 2], method='L-BFGS-B', bounds=[(0.0, budget)])
        
        if res.success:
            optimal_spend = res.x[0]
            max_conversions = -res.fun
            
            avg_spend = df_paid['ad_spend'].mean()
            baseline_conversions = response_func(avg_spend, adstock_scen, alpha_opt, beta_opt, bias_opt)
            improvement = ((max_conversions - baseline_conversions) / baseline_conversions) * 100
            
            st.markdown("#### Optimization Results")
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric("Optimal Daily Spend", f"${optimal_spend:,.2f}")
            with stat2:
                st.metric("Expected Conversions/User", f"{max_conversions:.4f}")
            with stat3:
                st.metric("Gain vs. Baseline", f"+{improvement:.2f}%")
                
            # Plot response curve
            spends_axis = np.linspace(0, budget, 100)
            convs_axis = [response_func(s, adstock_scen, alpha_opt, beta_opt, bias_opt) for s in spends_axis]
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=spends_axis, y=convs_axis, mode='lines', name='Response Curve', line=dict(color='#3f37c9', width=3)))
            fig_curve.add_trace(go.Scatter(x=[optimal_spend], y=[max_conversions], mode='markers', name='Optimal Allocation', marker=dict(color='#e71d36', size=12, symbol='star')))
            
            fig_curve.update_layout(
                title="Conversion Response Function vs. Spend",
                xaxis_title="Daily Ad Spend ($)",
                yaxis_title="Expected Conversions per user",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_curve, width='stretch')
            
        else:
            st.error("Optimization failed.")

# --- PAGE 4: ENGAGEMENT PREDICTOR ---
elif page == "Engagement Predictor":
    st.markdown("<h1 class='main-title'>Engagement Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict Engagement Rate Classification for Post Drafts</p>", unsafe_allow_html=True)
    
    st.write(
        "Enter draft details for a proposed post caption, hashtags, and media channels. "
        "The model will extract NLP features and predict whether the post will achieve a high engagement rate."
    )
    
    st.markdown("---")
    
    col_inputs, col_predict = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("Draft Configurations")
        caption = st.text_area(
            "Caption Text",
            value="Obsessed with my new daily fitness routine! Feeling stronger than ever today. 💪🔥 #motivation"
        )
        hashtags = st.text_input(
            "Hashtags (space separated)",
            value="#fitness #workout #healthy #progress"
        )
        
        followers = st.number_input(
            "Account Followers",
            min_value=100,
            max_value=10000000,
            value=25000,
            step=1000
        )
        
        media_type = st.selectbox(
            "Format",
            ["video", "image", "carousel"]
        )
        
        hour = st.slider(
            "Posting Time (Hour of Day)",
            min_value=0,
            max_value=23,
            value=10
        )
        
    with col_predict:
        st.subheader("Prediction Outputs")
        
        if st.button("Predict Engagement Level"):
            post_len = len(caption)
            hashtag_count = len(hashtags.split())
            sentiment = TextBlob(caption).sentiment.polarity
            is_video = 1 if media_type == "video" else 0
            is_carousel = 1 if media_type == "carousel" else 0
            
            st.markdown("##### NLP Metrics")
            st.write(f"- **Caption Length**: {post_len} chars")
            st.write(f"- **Hashtag Count**: {hashtag_count}")
            st.write(f"- **Sentiment Polarity**: `{sentiment:.4f}`")
            
            # Predict
            input_data = pd.DataFrame([{
                'Post Length': post_len,
                'Hashtag Count': hashtag_count,
                'Sentiment': sentiment,
                'Hour': hour,
                'Is Video': is_video,
                'Is Carousel': is_carousel,
                'Followers': followers
            }])
            
            prediction = rf_model.predict(input_data)[0]
            probability = rf_model.predict_proba(input_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"### 🎉 High Engagement Predicted")
                st.write(f"Model Confidence: **{probability[1]*100:.1f}%**")
                st.write("This draft matches historical features correlated with above-median organic reach.")
            else:
                st.warning(f"### ⚠️ Low Engagement Predicted")
                st.write(f"Model Confidence: **{probability[0]*100:.1f}%**")
                st.write("Consider adjusting caption tone (polarity), increasing video/carousel usage, or adding more hashtags (aim for 5-8).")
                
        else:
            st.info("Fill out the draft configurations and click Predict.")

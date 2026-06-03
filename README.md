<p align="center">
  <img src="assets/readme_banner.png" width="100%" alt="Instagram Analytics & Spend Optimizer Banner"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=yellow" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Pandas-v1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/SciPy-v1.8%2B-0C55A5?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/scikit--learn-v1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-v1.20%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Plotly-v5.10%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
</p>

---

# Instagram Content Engagement & Marketing Ad Campaign Optimization

This repository contains a python data science pipeline and interactive dashboard for Instagram analytics. The project covers organic post engagement prediction and paid advertising optimization.

---

## 🖥️ Streamlit Control Center Screenshots

### 1. Dashboard Overview
Provides an entry point introducing the dual-track functionality (organic analysis vs. paid budget solver).
<p align="center">
  <img src="assets/dashboard_overview.png" width="95%" alt="Dashboard Overview Tab"/>
</p>

### 2. Organic Performance Analysis
Visualizes post interaction distributions, follower engagement rates, and format-specific KPIs.
<p align="center">
  <img src="assets/organic_performance.png" width="95%" alt="Organic Performance Tab"/>
</p>

### 3. Paid Budget Optimizer
Estimates conversion response curves incorporating historical Adstock carryover and runs the live SciPy solver to calculate the optimal ad spend.
<p align="center">
  <img src="assets/paid_optimizer.png" width="95%" alt="Paid Budget Optimizer Tab"/>
</p>

### 4. AI Post Engagement Predictor
Runs a supervised Random Forest classifier to score caption draft parameters (lengths, formats, text sentiment polarity, hashtag counts) to predict organic reach.
<p align="center">
  <img src="assets/engagement_predictor.png" width="95%" alt="Engagement Predictor Tab"/>
</p>

---

## 🛠️ Repository Structure

*   `Analysis/Social_media_analytics.ipynb`: Jupyter notebook covering data cleaning, EDA, ML models, and SciPy budget optimization.
*   `Analysis/instagram_campaign_dataset.csv`: Organic posts dataset.
*   `Analysis/marketing_analytics_dataset.csv`: Paid ad spend and conversions dataset.
*   `app.py`: Streamlit web dashboard.
*   `requirements.txt`: Python package requirements.
*   `.gitignore`: Git exclusions.
*   `assets/`: Project banner and dashboard screenshots.

---

## 📈 Methodology & Mathematical Formulations

### 1. Adstock Carryover Effect
Advertising campaigns build brand memory over time. Cumulative adstock at day $t$ is calculated as:
$$A_t = S_t + \lambda A_{t-1}$$
where $S_t$ is the daily ad spend and $\lambda \in [0, 1)$ represents the decay parameter (retention rate).

### 2. Logarithmic Saturation Curve (Diminishing Returns)
To model marketing saturation (where doubling ad spend yields diminishing returns), we fit a logarithmic response curve to conversions:
$$\text{Expected Conversions} = \alpha \cdot \ln(x_{\text{ad\_spend}} + 1) + \beta \cdot A_{\text{adstock}} + C_{0}$$
The coefficients are estimated using least squares regression.

### 3. Constrained Budget Allocation Optimization
To find the optimal ad spend $x$ that maximizes conversions under a daily budget limit $B$, we solve:
$$\max_{x} \quad \alpha \cdot \ln(x + 1) + \beta \cdot A_{\text{adstock}} + C_{0}$$
$$\text{subject to} \quad 0 \le x \le B$$
This constrained optimization problem is solved using the `L-BFGS-B` algorithm via `scipy.optimize.minimize`.

---

## 🚀 Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Open the Notebook**:
   ```bash
   jupyter notebook Analysis/Social_media_analytics.ipynb
   ```

3. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

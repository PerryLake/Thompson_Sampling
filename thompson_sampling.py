import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pathlib

#------------------------------------------------------------------------------
# SECTION 1: CONFIGURATION AND SETUP
#------------------------------------------------------------------------------

# Create .streamlit directory and config.toml with color theme settings
def setup_streamlit_config():
    home_dir = pathlib.Path.home()
    
    streamlit_dir = home_dir / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    

    config_content = """
[theme]
base = "light"

primaryColor = "#0E2841"

backgroundColor = "#FFFAEC"

secondaryBackgroundColor = "#F2E8CF"

textColor = "#0E2841"

font = "sans serif"
"""
    
    config_path = streamlit_dir / "config.toml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return f"Configuration file created at: {config_path}"

setup_streamlit_config()


st.set_page_config(
    page_title="Promotion Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

#------------------------------------------------------------------------------
# SECTION 2: COLOR SCHEME AND STYLING
#------------------------------------------------------------------------------

color_scheme = {
    'primary': '#0E2841',       
    'secondary': '#1A3A5F',      
    'tertiary': '#2A5980',        
    'quaternary': '#3D618B',      
    'quinary': '#5C7EA3',       
    'background': '#FFFAEC', 
    'card_bg': '#F2E8CF'
}

st.markdown("""
    <style>
    /* Base styling */
    .stApp {
        background-color: #FFFAEC;
        color: #0E2841;
        font-size: 1.3rem;
    }
    
    /* Ensuring minimum size for all text elements */
    .stMarkdown, .stText, label, .stButton, .stSelectbox, .stNumberInput, .stTextInput {
        font-size: min(1.3rem, 12px) !important;  /* Minimum of 12px */
    }
    
    /* Headers */
    .main-header {
        font-size: 3.4rem;
        font-weight: bold;
        color: #0E2841;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2.6rem;
        font-weight: bold;
        color: #0E2841;
        margin-top: 2.2rem;
        margin-bottom: 1.6rem;
    }
    
    /* Card styling */
    .card {
        background-color: #F2E8CF;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 8px rgba(14, 40, 65, 0.12);
        color: #0E2841;
        font-size: 1.3rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FFFAEC;
        color: #FFFAEC;
        font-size: 1.3rem;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1A3A5F;
    }
    
    /* Progress bar */
    .stProgress .st-eb {
        background-color: #0E2841;
    }
    
    /* Tab styling - Updated to hide icons and fix text color */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F2E8CF;
        color: #0E2841;
        font-size: 1.3rem;
        padding: 0.8rem 1.3rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] svg {
        display: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0E2841 !important;
        color: #FFFAEC !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #FFFAEC !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #F2E8CF;
        padding: 26px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(14, 40, 65, 0.1);
    }
    div[data-testid="stMetric"] > div:first-child {
        color: #0E2841;
        font-size: 1.5rem;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        color: #0E2841;
        font-size: 2.3rem;
        font-weight: 600;
    }
    
    /* Sidebar styling - Enhanced */
    section[data-testid="stSidebar"] {
        background-color: #F2E8CF !important;
        border-right: 1px solid rgba(14, 40, 65, 0.1);
    }
    section[data-testid="stSidebar"] > div {
        background-color: #F2E8CF !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 1.3rem;
        color: #0E2841;
    }
    section[data-testid="stSidebar"] h2 {
        color: #0E2841;
        font-size: 1.8rem;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(14, 40, 65, 0.2);
    }
    
    /* Number input styling */
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stNumberInput input {
        background-color: #E6D7B0 !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        border: 1px solid rgba(14, 40, 65, 0.2);
        border-radius: 6px;
        padding: 12px !important;
    }
    
    /* Text input styling */
    section[data-testid="stSidebar"] .stTextInput label {
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #E6D7B0 !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        border: 1px solid rgba(14, 40, 65, 0.2);
        border-radius: 6px;
        padding: 12px !important;
    }
    
    /* Form field labels */
    .sidebar-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0E2841;
        margin-bottom: 5px;
        margin-top: 10px;
    }
    
    /* Run simulation button styling */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #0E2841;
        color: #FFFAEC; 
        font-size: 1.5rem;
        padding: 1rem 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        margin-top: 20px;
        width: 100%;
        border: 2px solid #0E2841;
        transition: all 0.3s;
        box-shadow: 0 4px 8px rgba(14, 40, 65, 0.25);
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1A3A5F;
        border-color: #1A3A5F;
        transform: translateY(-3px);
        box-shadow: 0 6px 10px rgba(14, 40, 65, 0.3);
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(14, 40, 65, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.4rem;
        color: #0E2841;
        background-color: #F2E8CF;
    }
    
    /* Text styling */
    p, li {
        font-size: 1.3rem;
        line-height: 1.8;
        color: #0E2841;
    }
    h3 {
        font-size: 2.2rem;
        color: #0E2841;
    }
    h4 {
        font-size: 1.7rem;
        color: #0E2841;
    }
    
    /* Table styling - Enhanced for comparison tables */
    .stDataFrame {
        font-size: 1.3rem;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame th, .stDataFrame td {
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        background-color: #FFFAEC !important;
    }
    .dataframe {
        background-color: #F2E8CF !important;
        border: none !important;
    }
    .dataframe th {
        background-color: #F2E8CF !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        text-align: left !important;
        padding: 15px !important;
        border: none !important;
    }
    .dataframe td {
        background-color: #FFFAEC !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        padding: 12px 15px !important;
        border: none !important;
        border-top: 1px solid rgba(14, 40, 65, 0.1) !important;
    }
    .css-1ht1j8y {
        background-color: #F2E8CF !important;  /* Table header row */
    }
    .css-fblp2m {
        color: #0E2841 !important;  /* Table row text */
    }
    .css-12w0qpk {
        background-color: #FFFAEC !important;  /* Table cells */
    }
    
    /* Specific styling for the comparison table */
    .comparison-table th {
        background-color: #F2E8CF !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
    }
    .comparison-table td {
        background-color: #FFFAEC !important;
        color: #0E2841 !important;
        font-size: 1.3rem !important;
    }
    
    /* Info boxes */
    .stAlert div {
        color: #0E2841 !important;
        font-size: 1.3rem !important;
        background-color: #F2E8CF !important;
    }
    
    /* Tooltip */
    .stTooltipIcon {
        color: #0E2841;
        font-size: 1.3rem;
    }
    
    /* Form inputs */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #0E2841 !important;
        font-size: 1.3rem !important;
    }
    
    /* Chart elements - Enhanced for better title display */
    .main-svg text {
        fill: #0E2841 !important;
        font-size: 14px !important;
    }
    .main-svg .legend text {
        font-size: 16px !important;
    }
    .main-svg .xtick text, .main-svg .ytick text {
        font-size: 14px !important;
    }
    .main-svg .gtitle {
        font-size: 22px !important;
        font-weight: bold !important;
        fill: #0E2841 !important;
    }
    .main-svg .xtitle, .main-svg .ytitle {
        font-size: 16px !important;
        fill: #0E2841 !important;
        font-weight: 600 !important;
    }
    
    /* Text color correction for elements with blue backgrounds */
    [style*="background-color: #0E2841"] {
        color: #FFFAEC !important;
    }
    [style*="background-color: #1A3A5F"] {
        color: #FFFAEC !important;
    }
    [style*="background-color: #2A5980"] {
        color: #FFFAEC !important;
    }
    
    /* Footer */
    .footer {
        font-size: 1.3rem;
        margin-top: 60px; 
        padding: 25px;
        color: #0E2841;
        text-align: center;
        background-color: #F2E8CF;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-header'>Promotion Strategy Analysis Dashboard</div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------
# SECTION 3: DATA LOADING AND PREPARATION
#------------------------------------------------------------------------------

@st.cache_data
def load_data():

    data = {
        'promotion_id': [0, 1, 2, 3, 4, 5, 6, 7],
        'conversion_rate': [0.027, 0.054, 0.112, 0.044, 0.091, 0.024, 0.042, 0.048],
        'acquisition_cost': [19, 66, 428, 165, 386, 261, 499, 105],
        'customer_lifetime': [27.38, 27.38, 27.38, 27.38, 27.38, 27.38, 27.38, 27.38],
        'profit_per_month': [15.94, 9.96, 15.64, 13.31, 20.42, 19.84, 18.92, 21.96]
    }
    df = pd.DataFrame(data)
    return df


df = load_data()


df['avg_lifetime_profit'] = (df['profit_per_month'] * df['customer_lifetime']) - df['acquisition_cost']
df['avg_lifetime_profit'] = df['avg_lifetime_profit'].round(2)

#------------------------------------------------------------------------------
# SECTION 4: SIDEBAR CONTROLS
#------------------------------------------------------------------------------


st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Promotion: Thompson Sampling</h2>", unsafe_allow_html=True)


number_of_customers = st.sidebar.number_input(
    "Customers",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
    format="%d"
)

seed_value = 42

st.sidebar.markdown("<h3 style='margin-top: 25px; margin-bottom: 15px;'>Custom Promotion</h3>", unsafe_allow_html=True)

add_custom = st.sidebar.checkbox("Add custom promotion")

if add_custom:

    st.sidebar.markdown("<p class='sidebar-label'>ID</p>", unsafe_allow_html=True)
    custom_id = st.sidebar.text_input("", value="8", key="custom_id", label_visibility="collapsed")
    

    st.sidebar.markdown("<p class='sidebar-label'>Conversion Rate (%)</p>", unsafe_allow_html=True)
    custom_conv_rate = st.sidebar.text_input("", value="5", key="custom_conv_rate", label_visibility="collapsed")

    st.sidebar.markdown("<p class='sidebar-label'>Acquisition Cost ($)</p>", unsafe_allow_html=True)
    custom_acq_cost = st.sidebar.text_input("", value="100", key="custom_acq_cost", label_visibility="collapsed")
    

    st.sidebar.markdown("<p class='sidebar-label'>Profit per Month ($)</p>", unsafe_allow_html=True)
    custom_profit = st.sidebar.text_input("", value="20.00", key="custom_profit", label_visibility="collapsed")


run_simulation = st.sidebar.button(
    "Run Simulation", 
    type="primary",
    use_container_width=True
)


if add_custom and run_simulation:
    try:

        new_promotion = {
            'promotion_id': int(custom_id),
            'conversion_rate': float(custom_conv_rate) / 100, 
            'acquisition_cost': float(custom_acq_cost),
            'customer_lifetime': df['customer_lifetime'].iloc[0],  
            'profit_per_month': float(custom_profit)
        }
        

        if int(custom_id) in df['promotion_id'].values:
            idx = df.index[df['promotion_id'] == int(custom_id)].tolist()[0]
            for key, value in new_promotion.items():
                df.at[idx, key] = value
        else:

            new_df = pd.DataFrame([new_promotion])
            df = pd.concat([df, new_df], ignore_index=True)

        idx = df.index[df['promotion_id'] == int(custom_id)].tolist()[0]
        df.at[idx, 'avg_lifetime_profit'] = (df.at[idx, 'profit_per_month'] * df.at[idx, 'customer_lifetime']) - df.at[idx, 'acquisition_cost']
        df.at[idx, 'avg_lifetime_profit'] = round(df.at[idx, 'avg_lifetime_profit'], 2)
    except ValueError:
        st.sidebar.error("Please enter valid numeric values for all fields")


st.sidebar.markdown("<hr style='margin-top: 40px; margin-bottom: 20px; border-color: rgba(14, 40, 65, 0.2);'>", unsafe_allow_html=True)
with st.sidebar.expander("About Parameters"):
    st.markdown("""
    - **Number of Customers**: The total number of simulated customer interactions.
    - **Custom Promotion**: Add your own promotion strategy with specific parameters.
    
    Click **Run Simulation** to begin the analysis.
    """)

#------------------------------------------------------------------------------
# SECTION 5: THOMPSON SAMPLING ALGORITHM
#------------------------------------------------------------------------------
def run_thompson_sampling():
    print("\nInitial DataFrame:")
    print(df[['promotion_id', 'conversion_rate', 'profit_per_month', 'acquisition_cost', 'avg_lifetime_profit']].to_string())
    

    np.random.seed(42)
    

    n_promotions = df.shape[0]
    n_successes = [0] * n_promotions
    n_failures = [0] * n_promotions
    total_company_profit = 0
    promotion_selected = [0] * n_promotions
    

    progress_bar = st.progress(0)
    batch_size = max(1, number_of_customers // 100)
    

    for customer in range(number_of_customers):
        sampled_theta = []
        profit_per_trial = []


        for i in range(n_promotions):
            total_profit = n_successes[i] * df.loc[i, 'avg_lifetime_profit']
            total_trials = n_successes[i] + n_failures[i]
            if total_trials > 0:
                profit_trial = total_profit / total_trials
            else:
                profit_trial = 0
            profit_per_trial.append(profit_trial)


        overall_avg_profit_per_trial = sum(profit_per_trial) / n_promotions


        for i in range(n_promotions):
            a = profit_per_trial[i] + 1
            b = overall_avg_profit_per_trial + 1
            theta = np.random.beta(a, b)
            sampled_theta.append(theta)


        chosen_promotion = np.argmax(sampled_theta)
        promotion_selected[chosen_promotion] += 1


        conversion_rate = df.loc[chosen_promotion, 'conversion_rate']
        did_convert = np.random.rand() < conversion_rate

        if did_convert:
            n_successes[chosen_promotion] += 1
            total_company_profit += df.loc[chosen_promotion, 'avg_lifetime_profit']
        else:
            n_failures[chosen_promotion] += 1
            

        if customer % batch_size == 0 or customer == number_of_customers - 1:
            progress_bar.progress((customer + 1) / number_of_customers)
    

    results_df = df.copy()
    results_df['successes'] = n_successes
    results_df['failures'] = n_failures
    results_df['total_profit'] = results_df['successes'] * results_df['avg_lifetime_profit']
    results_df['profit_per_trial'] = results_df['total_profit'] / (results_df['successes'] + results_df['failures'])


    print("\nPromotion Results After Simulation:")
    print(results_df.to_string())

    np.random.seed(42)
    total_profit_random_sampling = 0
    random_successes = [0] * n_promotions
    random_failures = [0] * n_promotions
    
    for customer in range(number_of_customers):

        index_of_promotion_to_try = np.random.randint(0, n_promotions)


        if np.random.rand() <= df.loc[index_of_promotion_to_try, 'conversion_rate']:
            random_successes[index_of_promotion_to_try] += 1
            total_profit_random_sampling += df.loc[index_of_promotion_to_try, 'avg_lifetime_profit']
        else:
            random_failures[index_of_promotion_to_try] += 1

    best_promo_idx = results_df['total_profit'].idxmax()
    best_promo_id = results_df.iloc[best_promo_idx]['promotion_id']
    

    print(f"\nBest Promotion ID: {best_promo_id}")
    print(f"Best Promotion Total Profit: ${results_df.iloc[best_promo_idx]['total_profit']:,.2f}")
    

    results = pd.DataFrame({
        'Promotion ID': results_df['promotion_id'].values,
        'Total Trials': [n_successes[i] + n_failures[i] for i in range(n_promotions)],
        'Successes': n_successes,
        'Failures': n_failures,
        'Conversion Rate': [n_successes[i]/(n_successes[i] + n_failures[i]) if (n_successes[i] + n_failures[i]) > 0 else 0 for i in range(n_promotions)],
        'Total Profit': results_df['total_profit'].values,
        'Profit Per Trial': results_df['profit_per_trial'].values
    })
    

    most_profitable_promotion = results[results['Promotion ID'] == best_promo_id].iloc[0]
    

    random_results = pd.DataFrame({
        'Promotion ID': df['promotion_id'].values,
        'Total Trials': [random_successes[i] + random_failures[i] for i in range(n_promotions)],
        'Successes': random_successes,
        'Failures': random_failures,
        'Conversion Rate': [random_successes[i]/(random_successes[i] + random_failures[i]) if (random_successes[i] + random_failures[i]) > 0 else 0 for i in range(n_promotions)],
        'Total Profit': [random_successes[i] * df.loc[i, 'avg_lifetime_profit'] for i in range(n_promotions)],
        'Profit Per Trial': [(random_successes[i] * df.loc[i, 'avg_lifetime_profit'])/(random_successes[i] + random_failures[i]) if (random_successes[i] + random_failures[i]) > 0 else 0 for i in range(n_promotions)]
    })
    

    comparison = pd.DataFrame({
        'Metric': ['Total Profit', 'Average Profit Per Promotion', 'Improvement (%)'],
        'Thompson Sampling': [
            f"${total_company_profit:,.2f}",
            f"${total_company_profit / n_promotions:,.2f}",
            f"{((total_company_profit / total_profit_random_sampling) - 1) * 100:.2f}%"
        ],
        'Random Sampling': [
            f"${total_profit_random_sampling:,.2f}",
            f"${total_profit_random_sampling / n_promotions:,.2f}",
            "Baseline"
        ]
    })
    
    return results, random_results, comparison, most_profitable_promotion, total_company_profit, total_profit_random_sampling
#------------------------------------------------------------------------------
# SECTION 6: SIMULATION EXECUTION
#------------------------------------------------------------------------------


if run_simulation:
    with st.spinner('Running Thompson Sampling simulation...'):
        thompson_results, random_results, comparison, most_profitable_promotion, thompson_total, random_total = run_thompson_sampling()
    st.session_state['results'] = thompson_results
    st.session_state['random_results'] = random_results
    st.session_state['comparison'] = comparison
    st.session_state['most_profitable'] = most_profitable_promotion
    st.session_state['thompson_total'] = thompson_total
    st.session_state['random_total'] = random_total
    st.success(f"Simulation completed for {number_of_customers:,} customers!")

has_results = 'results' in st.session_state

#------------------------------------------------------------------------------
# SECTION 7: TABS AND VISUALIZATION
#------------------------------------------------------------------------------


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Thompson Sampling Results", "Comparison Analysis", "Recommendation"])


# Tab 1: Overview
with tab1:
    st.markdown("<div class='sub-header'>Promotion Data Overview</div>", unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            df,
            x='promotion_id',
            y='conversion_rate',
            title='Conversion Rate by Promotion',
            labels={'promotion_id': 'Promotion ID', 'conversion_rate': 'Conversion Rate'},
            color='conversion_rate',
            color_continuous_scale=[color_scheme['primary'], color_scheme['secondary'], color_scheme['tertiary'], 
                                   color_scheme['quaternary'], color_scheme['quinary']]
        )
        fig.update_layout(
            xaxis_title="Promotion ID", 
            yaxis_title="Conversion Rate",
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16,
            title_font_size=20
        )
        fig.update_traces(marker_line_color=color_scheme['primary'], marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            df,
            x='promotion_id',
            y='avg_lifetime_profit',
            title='Average Lifetime Profit per Promotion',
            labels={'promotion_id': 'Promotion ID', 'avg_lifetime_profit': 'Avg Lifetime Profit ($)'},
            color='avg_lifetime_profit',
            color_continuous_scale=[color_scheme['primary'], color_scheme['secondary'], color_scheme['tertiary']]
        )
        fig.update_layout(
            xaxis_title="Promotion ID", 
            yaxis_title="Avg Lifetime Profit ($)",
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16,
            title_font_size=20
        )
        fig.update_traces(marker_line_color=color_scheme['primary'], marker_line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.scatter(
            df,
            x='conversion_rate',
            y='avg_lifetime_profit',
            title='Avg Lifetime Profit vs Conversion Rate',
            labels={'conversion_rate': 'Conversion Rate', 'avg_lifetime_profit': 'Avg Lifetime Profit ($)'},
            size='customer_lifetime',
            color='promotion_id',
            text='promotion_id',
            color_discrete_sequence=[
                color_scheme['primary'], color_scheme['secondary'], color_scheme['tertiary'],
                color_scheme['quaternary'], color_scheme['quinary'], '#1A3A5F', '#2A5980', '#3D618B'
            ]
        )
        fig.update_traces(textposition='top center', marker=dict(sizemode='area', sizeref=0.1))
        fig.update_layout(
            xaxis_title="Conversion Rate", 
            yaxis_title="Avg Lifetime Profit ($)",
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='sub-header'>Thompson Sampling Results</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see results here.")
    else:

        results = st.session_state['results']
        most_profitable = st.session_state['most_profitable']

        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{number_of_customers:,}")
        with col2:
            st.metric("Best Promotion", f"Promotion {most_profitable['Promotion ID']}")
        with col3:
            st.metric("Best Promotion Profit", f"${most_profitable['Total Profit']:,.2f}")
        with col4:
            st.metric("Conversion Rate", f"{most_profitable['Conversion Rate']:.2%}")
        

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Thompson Sampling Results Table")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        

        st.markdown("<div class='sub-header'>Visualizations</div>", unsafe_allow_html=True)
        

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Trials per Promotion', 'Successes per Promotion', 'Total Profit per Promotion')
        )
        
        fig.add_trace(
            go.Bar(x=results['Promotion ID'], y=results['Total Trials'], name='Total Trials', 
                   marker_color=color_scheme['primary'], marker_line_color=color_scheme['secondary'], marker_line_width=1),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=results['Promotion ID'], y=results['Successes'], name='Successes', 
                   marker_color=color_scheme['secondary'], marker_line_color=color_scheme['primary'], marker_line_width=1),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=results['Promotion ID'], y=results['Total Profit'], name='Total Profit', 
                   marker_color=color_scheme['tertiary'], marker_line_color=color_scheme['primary'], marker_line_width=1),
            row=1, col=3
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Thompson Sampling Results by Promotion",
            title_font=dict(size=22, color=color_scheme['primary']),
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16
        )
        
        fig.update_xaxes(title_text="Promotion ID", title_font=dict(size=16, color=color_scheme['primary']))
        fig.update_yaxes(title_text="Count", row=1, col=1, title_font=dict(size=16, color=color_scheme['primary']))
        fig.update_yaxes(title_text="Count", row=1, col=2, title_font=dict(size=16, color=color_scheme['primary']))
        fig.update_yaxes(title_text="Profit ($)", row=1, col=3, title_font=dict(size=16, color=color_scheme['primary']))
        
        st.plotly_chart(fig, use_container_width=True)
        

        fig = px.scatter(
            results,
            x='Conversion Rate',
            y='Profit Per Trial',
            size='Total Trials',
            color='Total Profit',
            hover_name='Promotion ID',
            size_max=60,
            title='Conversion Rate vs. Profit Per Trial',
            color_continuous_scale=[
                color_scheme['primary'], color_scheme['secondary'], color_scheme['tertiary'],
                color_scheme['quaternary'], color_scheme['quinary']
            ]
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Conversion Rate",
            yaxis_title="Profit Per Trial ($)",
            coloraxis_colorbar_title="Total Profit",
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16,
            title_font_size=22
        )
        
        st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.markdown("<div class='sub-header'>Thompson vs Random Sampling Comparison</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see comparison here.")
    else:

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Thompson Sampling: Promotion Distribution")

            fig = px.bar(
                thompson_results.sort_values('Total Trials', ascending=True),
                y='Promotion ID',
                x='Total Trials',
                title='Promotion Distribution with Thompson Sampling',
                orientation='h',
                color='Total Profit',
                color_continuous_scale=[
                    color_scheme['primary'], color_scheme['secondary'], color_scheme['tertiary'],
                    color_scheme['quaternary'], color_scheme['quinary']
                ]
            )
            fig.update_layout(
                plot_bgcolor=color_scheme['background'],
                paper_bgcolor=color_scheme['background'],
                font_color=color_scheme['primary'],
                font_size=16,
                title_font_size=18,
                yaxis=dict(title='Promotion ID', type='category'),
                xaxis=dict(title='Total Trials')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Random Sampling: Promotion Distribution")

            fig = px.bar(
                random_results.sort_values('Total Trials', ascending=True),
                y='Promotion ID',
                x='Total Trials',
                title='Promotion Distribution with Random Sampling',
                orientation='h',
                color='Total Profit',
                color_continuous_scale=[
                    color_scheme['secondary'], color_scheme['tertiary'], color_scheme['quaternary'],
                    color_scheme['quinary'], '#1A3A5F'
                ]
            )
            fig.update_layout(
                plot_bgcolor=color_scheme['background'],
                paper_bgcolor=color_scheme['background'],
                font_color=color_scheme['primary'],
                font_size=16,
                title_font_size=18,
                yaxis=dict(title='Promotion ID', type='category'),
                xaxis=dict(title='Total Trials')
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        

        combined_profit_data = pd.DataFrame({
            'Promotion ID': thompson_results['Promotion ID'],
            'Thompson Profit': thompson_results['Total Profit'],
            'Random Profit': random_results['Total Profit']
        })

        profit_fig = px.bar(
            combined_profit_data,
            x='Promotion ID',
            y=['Thompson Profit', 'Random Profit'],
            title='Profit by Promotion: Thompson vs Random',
            barmode='group',
            labels={'value': 'Total Profit ($)', 'variable': 'Method'},
            color_discrete_map={
                'Thompson Profit': color_scheme['primary'],
                'Random Profit': color_scheme['quinary']
            }
        )

        profit_fig.update_layout(
            height=500,
            xaxis_title="Promotion ID",
            yaxis_title="Total Profit ($)",
            legend_title="Method",
            plot_bgcolor=color_scheme['background'],
            paper_bgcolor=color_scheme['background'],
            font_color=color_scheme['primary'],
            font_size=16,
            title_font_size=22
        )
        profit_fig.update_traces(marker_line_color=color_scheme['primary'], marker_line_width=1)
        
        st.plotly_chart(profit_fig, use_container_width=True)


with tab4:
    st.markdown("<div class='sub-header'>Promotion Recommendation</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see recommendations here.")
    else:
        most_profitable = st.session_state['most_profitable']
        promo_id = int(most_profitable['Promotion ID'])
        
        promo_details = df[df['promotion_id'] == promo_id].iloc[0]
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader(f"Recommended Promotion: Promotion {promo_id}")
        
        st.write(f"Based on Thompson Sampling with {number_of_customers:,} customers, "
                f"we recommend implementing **Promotion {promo_id}** for optimal results.")

        st.markdown("#### Key Performance Metrics:")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Total Profit Generated:** ${most_profitable['Total Profit']:,.2f}")
            st.markdown(f"**Conversion Rate:** {most_profitable['Conversion Rate']:.2%}")
            st.markdown(f"**Profit Per Trial:** ${most_profitable['Profit Per Trial']:,.2f}")
            
        with col2:
            st.markdown(f"**Total Trials:** {int(most_profitable['Total Trials']):,}")
            st.markdown(f"**Total Conversions:** {int(most_profitable['Successes']):,}")
        
        st.markdown("#### Promotion Details:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Acquisition Cost:** ${promo_details['acquisition_cost']:.2f}")
            st.markdown(f"**Monthly Profit per Customer:** ${promo_details['profit_per_month']:.2f}")
            
        with col2:
            st.markdown(f"**Customer Lifetime:** {promo_details['customer_lifetime']:.2f} months")
            st.markdown(f"**Avg Lifetime Profit:** ${promo_details['avg_lifetime_profit']:.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
     


#------------------------------------------------------------------------------
# SECTION 8: FOOTER
#------------------------------------------------------------------------------

# Footer with updated text
st.markdown("""
    <div class="footer">
        <p>Project 1: Thompson Sampling Promotion</p>
    </div>
""", unsafe_allow_html=True)




# These packages should be installed via terminal before running:
# pip install matplotlib pandas plotly numpy streamlit

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
    # Get the home directory
    home_dir = pathlib.Path.home()
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = home_dir / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml with theme settings
    config_content = """
[theme]
# Use the light theme as base
base = "light"

# Deep navy blue primary color for interactive elements
primaryColor = "#0E2841"

# Soft cream background color
backgroundColor = "#FFFAEC"

# Subtle cream variant for sidebar and widgets
secondaryBackgroundColor = "#F2E8CF"

# Text color matching primary navy
textColor = "#0E2841"

# Font family - sans serif works well with this design
font = "sans serif"
"""
    
    # Write the config file
    config_path = streamlit_dir / "config.toml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return f"Configuration file created at: {config_path}"

# Set up the Streamlit configuration
# setup_streamlit_config()

# Set page configuration
st.set_page_config(
    page_title="Promotion Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

#------------------------------------------------------------------------------
# SECTION 2: COLOR SCHEME AND STYLING
#------------------------------------------------------------------------------

# Custom color scheme for all visualizations
color_scheme = {
    'primary': '#0E2841',         # Deep navy blue
    'secondary': '#1A3A5F',       # Lighter navy
    'tertiary': '#2A5980',        # Even lighter navy
    'quaternary': '#3D618B',      # Muted navy-blue
    'quinary': '#5C7EA3',         # Lighter blue
    'background': '#FFFAEC',      # Soft cream
    'card_bg': '#F2E8CF'          # Subtle cream variant
}

# Add custom CSS for styling (large section - collapsed for readability)
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
        background-color: #0E2841;
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

# Main title
st.markdown("<div class='main-header'>Promotion Strategy Analysis Dashboard</div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------
# SECTION 3: DATA LOADING AND PREPARATION
#------------------------------------------------------------------------------

# Function to load data
@st.cache_data
def load_data():
    # Use the provided promotion data
    data = {
        'promotion_id': [0, 1, 2, 3, 4, 5, 6, 7],
        'conversion_rate': [0.027, 0.054, 0.112, 0.044, 0.091, 0.024, 0.042, 0.048],
        'acquisition_cost': [19, 66, 428, 165, 386, 261, 499, 105],
        'customer_lifetime': [27.38, 27.38, 27.38, 27.38, 27.38, 27.38, 27.38, 27.38],
        'profit_per_month': [15.94, 9.96, 15.64, 13.31, 20.42, 19.84, 18.92, 21.96]
    }
    df = pd.DataFrame(data)
    return df

# Load data
df = load_data()

# Calculate avg_lifetime_profit
df['avg_lifetime_profit'] = (df['profit_per_month'] * df['customer_lifetime']) - df['acquisition_cost']
df['avg_lifetime_profit'] = df['avg_lifetime_profit'].round(2)

#------------------------------------------------------------------------------
# SECTION 4: SIDEBAR CONTROLS
#------------------------------------------------------------------------------

# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Promotion: Thompson Sampling</h2>", unsafe_allow_html=True)

# Number of customers input
number_of_customers = st.sidebar.number_input(
    "Customers",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000,
    format="%d"
)

# Set a fixed seed value for reproducibility (hidden from UI)
seed_value = 42

# Custom promotion section with matching style to header
st.sidebar.markdown("<h3 style='margin-top: 25px; margin-bottom: 15px;'>Custom Promotion</h3>", unsafe_allow_html=True)

# Form for adding a custom promotion
add_custom = st.sidebar.checkbox("Add custom promotion")

if add_custom:
    # ID field
    st.sidebar.markdown("<p class='sidebar-label'>ID</p>", unsafe_allow_html=True)
    custom_id = st.sidebar.text_input("", value="8", key="custom_id", label_visibility="collapsed")
    
    # Conversion Rate field
    st.sidebar.markdown("<p class='sidebar-label'>Conversion Rate (%)</p>", unsafe_allow_html=True)
    custom_conv_rate = st.sidebar.text_input("", value="5", key="custom_conv_rate", label_visibility="collapsed")
    
    # Acquisition Cost field
    st.sidebar.markdown("<p class='sidebar-label'>Acquisition Cost ($)</p>", unsafe_allow_html=True)
    custom_acq_cost = st.sidebar.text_input("", value="100", key="custom_acq_cost", label_visibility="collapsed")
    
    # Profit per Month field
    st.sidebar.markdown("<p class='sidebar-label'>Profit per Month ($)</p>", unsafe_allow_html=True)
    custom_profit = st.sidebar.text_input("", value="20.00", key="custom_profit", label_visibility="collapsed")

# Run button
run_simulation = st.sidebar.button(
    "Run Simulation", 
    type="primary",
    use_container_width=True
)

# Add custom promotion to dataframe if checked
if add_custom and run_simulation:
    try:
        # Parse custom promotion values
        new_promotion = {
            'promotion_id': int(custom_id),
            'conversion_rate': float(custom_conv_rate) / 100,  # Convert from percentage to decimal
            'acquisition_cost': float(custom_acq_cost),
            'customer_lifetime': df['customer_lifetime'].iloc[0],  # Use same lifetime as other promotions
            'profit_per_month': float(custom_profit)
        }
        
        # Check if this ID already exists
        if int(custom_id) in df['promotion_id'].values:
            # Update existing row
            idx = df.index[df['promotion_id'] == int(custom_id)].tolist()[0]
            for key, value in new_promotion.items():
                df.at[idx, key] = value
        else:
            # Create a new single-row DataFrame and concatenate with existing df
            new_df = pd.DataFrame([new_promotion])
            df = pd.concat([df, new_df], ignore_index=True)
        
        # Recalculate avg_lifetime_profit for the new/updated promotion
        idx = df.index[df['promotion_id'] == int(custom_id)].tolist()[0]
        df.at[idx, 'avg_lifetime_profit'] = (df.at[idx, 'profit_per_month'] * df.at[idx, 'customer_lifetime']) - df.at[idx, 'acquisition_cost']
        df.at[idx, 'avg_lifetime_profit'] = round(df.at[idx, 'avg_lifetime_profit'], 2)
    except ValueError:
        st.sidebar.error("Please enter valid numeric values for all fields")

# Info section
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
    """Run Thompson Sampling algorithm exactly matching the Tyler_585_project_1.ipynb implementation"""
    # Print the initial values to ensure data is correct
    print("\nInitial DataFrame:")
    print(df[['promotion_id', 'conversion_rate', 'profit_per_month', 'acquisition_cost', 'avg_lifetime_profit']].to_string())
    
    # Use the same random seed as the notebook
    np.random.seed(42)
    
    # Initialize trackers - exactly as in the notebook
    n_promotions = df.shape[0]
    n_successes = [0] * n_promotions
    n_failures = [0] * n_promotions
    total_company_profit = 0
    promotion_selected = [0] * n_promotions
    
    # Create progress bar
    progress_bar = st.progress(0)
    batch_size = max(1, number_of_customers // 100)
    
    # Run simulation
    for customer in range(number_of_customers):
        sampled_theta = []
        profit_per_trial = []

        # Compute actual profit per trial - exactly as in the notebook
        for i in range(n_promotions):
            total_profit = n_successes[i] * df.loc[i, 'avg_lifetime_profit']
            total_trials = n_successes[i] + n_failures[i]
            if total_trials > 0:
                profit_trial = total_profit / total_trials
            else:
                profit_trial = 0
            profit_per_trial.append(profit_trial)

        # Compute overall average profit per trial - exactly as in the notebook
        overall_avg_profit_per_trial = sum(profit_per_trial) / n_promotions

        # Draw samples from beta distribution - exactly as in the notebook
        for i in range(n_promotions):
            a = profit_per_trial[i] + 1
            b = overall_avg_profit_per_trial + 1
            theta = np.random.beta(a, b)
            sampled_theta.append(theta)

        # Choose promotion with highest theta - exactly as in the notebook
        chosen_promotion = np.argmax(sampled_theta)
        promotion_selected[chosen_promotion] += 1

        # Simulate conversion - exactly as in the notebook
        conversion_rate = df.loc[chosen_promotion, 'conversion_rate']
        did_convert = np.random.rand() < conversion_rate

        if did_convert:
            n_successes[chosen_promotion] += 1
            total_company_profit += df.loc[chosen_promotion, 'avg_lifetime_profit']
        else:
            n_failures[chosen_promotion] += 1
            
        # Update progress bar
        if customer % batch_size == 0 or customer == number_of_customers - 1:
            progress_bar.progress((customer + 1) / number_of_customers)
    
    # Update DataFrame with final results - exactly as in the notebook
    results_df = df.copy()
    results_df['successes'] = n_successes
    results_df['failures'] = n_failures
    results_df['total_profit'] = results_df['successes'] * results_df['avg_lifetime_profit']
    results_df['profit_per_trial'] = results_df['total_profit'] / (results_df['successes'] + results_df['failures'])

    # Print the results DataFrame to debug - same as notebook output
    print("\nPromotion Results After Simulation:")
    print(results_df.to_string())
    
    # Random sampling for comparison - exactly as in the notebook
    np.random.seed(42)
    total_profit_random_sampling = 0
    random_successes = [0] * n_promotions
    random_failures = [0] * n_promotions
    
    for customer in range(number_of_customers):
        # Choose a promotion at random
        index_of_promotion_to_try = np.random.randint(0, n_promotions)

        # Simulate if the promotion is successful
        if np.random.rand() <= df.loc[index_of_promotion_to_try, 'conversion_rate']:
            # Increment the total profit
            random_successes[index_of_promotion_to_try] += 1
            total_profit_random_sampling += df.loc[index_of_promotion_to_try, 'avg_lifetime_profit']
        else:
            random_failures[index_of_promotion_to_try] += 1
    
    # Find the promotion with the maximum total profit - critical for determining "best" promotion
    best_promo_idx = results_df['total_profit'].idxmax()
    best_promo_id = results_df.iloc[best_promo_idx]['promotion_id']
    
    # Print debug info about the best promotion
    print(f"\nBest Promotion ID: {best_promo_id}")
    print(f"Best Promotion Total Profit: ${results_df.iloc[best_promo_idx]['total_profit']:,.2f}")
    
    # Format for dashboard display
    results = pd.DataFrame({
        'Promotion ID': results_df['promotion_id'].values,
        'Total Trials': [n_successes[i] + n_failures[i] for i in range(n_promotions)],
        'Successes': n_successes,
        'Failures': n_failures,
        'Conversion Rate': [n_successes[i]/(n_successes[i] + n_failures[i]) if (n_successes[i] + n_failures[i]) > 0 else 0 for i in range(n_promotions)],
        'Total Profit': results_df['total_profit'].values,
        'Profit Per Trial': results_df['profit_per_trial'].values
    })
    
    # Ensure the best promotion is correctly identified in the results DataFrame
    most_profitable_promotion = results[results['Promotion ID'] == best_promo_id].iloc[0]
    
    # Random results for dashboard
    random_results = pd.DataFrame({
        'Promotion ID': df['promotion_id'].values,
        'Total Trials': [random_successes[i] + random_failures[i] for i in range(n_promotions)],
        'Successes': random_successes,
        'Failures': random_failures,
        'Conversion Rate': [random_successes[i]/(random_successes[i] + random_failures[i]) if (random_successes[i] + random_failures[i]) > 0 else 0 for i in range(n_promotions)],
        'Total Profit': [random_successes[i] * df.loc[i, 'avg_lifetime_profit'] for i in range(n_promotions)],
        'Profit Per Trial': [(random_successes[i] * df.loc[i, 'avg_lifetime_profit'])/(random_successes[i] + random_failures[i]) if (random_successes[i] + random_failures[i]) > 0 else 0 for i in range(n_promotions)]
    })
    
    # Comparison for dashboard
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


# Run simulation if button clicked
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
# Check if we have results
has_results = 'results' in st.session_state

#------------------------------------------------------------------------------
# SECTION 7: TABS AND VISUALIZATION
#------------------------------------------------------------------------------

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Thompson Sampling Results", "Comparison Analysis", "Recommendation"])


# Tab 1: Overview
with tab1:
    st.markdown("<div class='sub-header'>Promotion Data Overview</div>", unsafe_allow_html=True)
    
    # Display data overview with Plotly
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
    
    # Additional overview charts
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

# Tab 2: Thompson Sampling Results
with tab2:
    st.markdown("<div class='sub-header'>Thompson Sampling Results</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see results here.")
    else:
        # Show Thompson Sampling results
        results = st.session_state['results']
        most_profitable = st.session_state['most_profitable']
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{number_of_customers:,}")
        with col2:
            st.metric("Best Promotion", f"Promotion {most_profitable['Promotion ID']}")
        with col3:
            st.metric("Best Promotion Profit", f"${most_profitable['Total Profit']:,.2f}")
        with col4:
            st.metric("Conversion Rate", f"{most_profitable['Conversion Rate']:.2%}")
        
        # Results table
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Thompson Sampling Results Table")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create visualizations
        st.markdown("<div class='sub-header'>Visualizations</div>", unsafe_allow_html=True)
        
        # Subplot with Total Trials, Successes, and Total Profit
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
        
        # Conversion rate vs Profit per trial
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

# Tab 3: Comparison Analysis
with tab3:
    st.markdown("<div class='sub-header'>Thompson vs Random Sampling Comparison</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see comparison here.")
    else:
        # [Keep comparison table and bar chart comparison the same]

        # Additional comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Thompson Sampling: Promotion Distribution")
            # Horizontal bar chart
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
            # Also make this a horizontal bar chart (matching the Thompson chart)
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
        
        # Side-by-side profit comparison
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

# Tab 4: Recommendation
with tab4:
    st.markdown("<div class='sub-header'>Promotion Recommendation</div>", unsafe_allow_html=True)
    
    if not has_results:
        st.info("👈 Set simulation parameters and click 'Run Simulation' to see recommendations here.")
    else:
        # Get the most profitable promotion
        most_profitable = st.session_state['most_profitable']
        promo_id = int(most_profitable['Promotion ID'])
        
        # Get the promotion details from the original dataframe
        promo_details = df[df['promotion_id'] == promo_id].iloc[0]
        
        # Create a card for the recommendation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Header with trophy emoji
        st.subheader(f"Recommended Promotion: Promotion {promo_id}")
        
        # Description
        st.write(f"Based on Thompson Sampling with {number_of_customers:,} customers, "
                f"we recommend implementing **Promotion {promo_id}** for optimal results.")
        
        # Key metrics section
        st.markdown("#### Key Performance Metrics:")
        
        # Create two columns for the metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Total Profit Generated:** ${most_profitable['Total Profit']:,.2f}")
            st.markdown(f"**Conversion Rate:** {most_profitable['Conversion Rate']:.2%}")
            st.markdown(f"**Profit Per Trial:** ${most_profitable['Profit Per Trial']:,.2f}")
            
        with col2:
            st.markdown(f"**Total Trials:** {int(most_profitable['Total Trials']):,}")
            st.markdown(f"**Total Conversions:** {int(most_profitable['Successes']):,}")
        
        # Promotion details section
        st.markdown("#### Promotion Details:")
        
        # Create two columns for the details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Acquisition Cost:** ${promo_details['acquisition_cost']:.2f}")
            st.markdown(f"**Monthly Profit per Customer:** ${promo_details['profit_per_month']:.2f}")
            
        with col2:
            st.markdown(f"**Customer Lifetime:** {promo_details['customer_lifetime']:.2f} months")
            st.markdown(f"**Avg Lifetime Profit:** ${promo_details['avg_lifetime_profit']:.2f}")
        
        # Close the card div
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



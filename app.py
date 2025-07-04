# System
import os
from datetime import datetime
# Data

import numpy as np
import pandas as pd
import feedparser
from finvizfinance.news import News
# ML
from transformers import pipeline
# Frontend/Visuals
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Market Sentiment",
                   layout="wide",
                   page_icon=":chart_with_upwards_trend:")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .stApp {
#         max-width: 1000px;
#         margin: 0 auto;
#         padding: 2rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Settings
CSV_FILE = "sentiment_history.csv"

# Initialize sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=0  # use -1 if no GPU
    )

sentiment_analysis = load_sentiment_model()

def sentiment_score(text):
    return sentiment_analysis(text)[0]['label'] == "POSITIVE"

def get_finviz_news():
    fnews = News()
    all_news = fnews.get_news()
    news_df = pd.DataFrame(all_news['news'])
    return news_df['Title'].to_list()

def news_avg_sentiment():
    titles = get_finviz_news()
    scores = [sentiment_score(title) for title in titles]
    return np.mean(scores)

def calculate_z_score(current_score, historical_scores):
    mean = np.mean(historical_scores)
    std_dev = np.std(historical_scores)
    z_score = (current_score - mean) / std_dev if std_dev != 0 else 0  # Avoid division by zero
    return z_score

def calculate_quantile(current_score, historical_scores):
    quantile = np.percentile(historical_scores, current_score * 100)  # Converts score to percentage
    return quantile

def save_sentiment(average_sentiment):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({'timestamp': [now], 'average_sentiment': [average_sentiment]})

    if os.path.exists(CSV_FILE):
        new_entry.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(CSV_FILE, index=False)

def load_sentiment_history():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=['timestamp', 'average_sentiment'])

@st.cache_data(ttl=3600)  # cache the result for 2 hour (7200 seconds), re-runs automatically
def retrieve_sentiment():
    sentiment_history = load_sentiment_history()

    if sentiment_history.empty or (datetime.now() - pd.to_datetime(sentiment_history['timestamp'].iloc[-1])).total_seconds() > 7200:
        avg_sentiment = news_avg_sentiment()
        save_sentiment(avg_sentiment)
        sentiment_history = load_sentiment_history()
    else:
        avg_sentiment = sentiment_history['average_sentiment'].iloc[-1]

    return avg_sentiment, sentiment_history

def plot_sentiment_over_time(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['average_sentiment'],
        mode='lines+markers',
        line=dict(color='green'),
        marker=dict(size=8),
        name='Sentiment Score'
    ))

    fig.update_layout(
        #title="Sentiment Score Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Average News Sentiment",
        yaxis_range=[0, 1],
        template="plotly_white",
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_text(avg_sentiment):
    st.markdown(
        f"<h1 style='text-align: center; font-size: 100px;'>{avg_sentiment:.2f}</h1>",
        unsafe_allow_html=True
    )

def display_sentiment_fuel_gauge(avg_sentiment):
    # Create many small color steps to fake a gradient
    n_steps = 50
    colors = []
    for i in np.linspace(0, 1, n_steps):
        if i < 0.5:
            # interpolate red to yellow
            r = 255
            g = int(255 * (i / 0.5))
            b = 0
        else:
            # interpolate yellow to green
            r = int(255 * (1 - (i - 0.5) / 0.5))
            g = 255
            b = 0
        colors.append({'range': [i, i + (1/n_steps)], 'color': f'rgb({r},{g},{b})'})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_sentiment,
        number={'font': {'size': 100}, 'valueformat': '.2f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': colors,
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': avg_sentiment
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        #width=600,
        height=300,
    )

    # Center the gauge
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    avg_sentiment, sentiment_history = retrieve_sentiment()

    # Frontend
    # Background color based on sentiment
    red_intensity = int((1 - avg_sentiment) * 255)
    green_intensity = int(avg_sentiment * 255)
    background_color = f'rgb({red_intensity}, {green_intensity}, 100)'

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {background_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display current sentiment
    st.title("📰 Financial News Sentiment Index")

    display_sentiment_fuel_gauge(avg_sentiment)
    

    # Small text for last update
    last_updated = pd.to_datetime(sentiment_history['timestamp'].iloc[-1])
    st.markdown(
        f"<p style='text-align: center; font-size: 16px; color: white;'>Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True
    )

    #st.write("**Remark:** 1.0 = fully positive, 0.0 = fully negative")

    # Plot sentiment over time
    st.subheader("📈 Sentiment History")
    if not sentiment_history.empty:
        sentiment_history['timestamp'] = pd.to_datetime(sentiment_history['timestamp'])
        plot_sentiment_over_time(sentiment_history)
    
        # Download sentiment history as CSV
        st.download_button(
            label="📥 Download Sentiment Data",
            data=sentiment_history.to_csv(index=False).encode('utf-8'),
            file_name='sentiment_history.csv',
            mime='text/csv'
        )
        

if __name__ == "__main__":
    main()

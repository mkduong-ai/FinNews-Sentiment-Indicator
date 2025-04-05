# System
import os
from datetime import datetime
# Data
import pandas as pd
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

# Functions
def sentiment_score(text):
    return sentiment_analysis(text)[0]['label'] == "POSITIVE"

def news_avg_sentiment():
    fnews = News()
    all_news = fnews.get_news()
    news_df = pd.DataFrame(all_news['news'])
    news_df['sentiment_score'] = news_df['Title'].apply(sentiment_score)
    return news_df['sentiment_score'].mean()

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

@st.cache_data(ttl=7200)  # cache the result for 2 hour (7200 seconds), re-runs automatically
def retrieve_sentiment():
    sentiment_history = load_sentiment_history()

    if sentiment_history.empty or (datetime.now() - pd.to_datetime(sentiment_history['timestamp'].iloc[-1])).total_seconds() > 7190:
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
    st.title("📰 FinNews: Financial News Sentiment")

    st.markdown(
        f"<h1 style='text-align: center; font-size: 100px;'>{avg_sentiment:.2f}</h1>",
        unsafe_allow_html=True
    )

    # Small text for last update
    last_updated = pd.to_datetime(sentiment_history['timestamp'].iloc[-1])
    st.markdown(
        f"<p style='text-align: center; font-size: 16px; color: gray;'>Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True
    )

    st.write("**Remark:** 1.0 = fully positive, 0.0 = fully negative")

    # Plot sentiment over time
    st.subheader("📈 Sentiment History")
    if not sentiment_history.empty:
        sentiment_history['timestamp'] = pd.to_datetime(sentiment_history['timestamp'])
        plot_sentiment_over_time(sentiment_history)
        

if __name__ == "__main__":
    main()

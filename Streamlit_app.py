import boto3
import json
import pandas as pd
from openai import OpenAI
import streamlit as st
import os

bucket_name="loadforecastingdata"


access_key=os.getenv("access key")

aws_s3_key=os.getenv("secret api key aws")

openai_key = os.getenv("openai_key")


@st.cache_data
def load_csv_from_s3(path, access_key, secret_key, bucket_name):
    
    # # Read CSV from S3
    df = pd.read_csv(
        f"s3://{bucket_name}/{path}",
        storage_options={
            "key": access_key,
            "secret": secret_key,
            "client_kwargs": {
                "region_name": "eu-west-1"
            }
        }
    )
    
    return df


End = load_csv_from_s3("Data/End.csv", access_key, aws_s3_key, bucket_name)
Prev_completa = load_csv_from_s3("Data/Prev_completa.csv", access_key, aws_s3_key, bucket_name)



client = OpenAI(api_key=openai_key)


# Normalize Date column once
End["Date"] = pd.to_datetime(End["Date"])
def shap_return(date: str,aggregation:str) -> dict:
    # # Convert string to datetime
    date = pd.to_datetime(date)

    # # # Determine aggregation level
    # if date.hour == 0 and date.minute == 0:
    #     aggregation = "daily"
    # elif date.minute == 0:
    #     aggregation = "hourly"
    # else:
    #     aggregation = "quarter_hour"

    if aggregation == "quarter_hour":
        row = End.loc[End["Date"] == date]

        if row.empty:
            return {"error": "Date not found"}

        forecast = row["Forecasted"].iloc[0]

        shap_values = (
            row.drop(columns=["Forecasted", "Date"])
            .iloc[0]
            .to_dict()
        )

        return {
            "aggregation": aggregation,
            "forecast": float(forecast),
            "Shapley": shap_values
        }

    elif aggregation == "hourly":
        mask = (
            (End["Date"].dt.date == date.date()) &
            (End["Date"].dt.hour == date.hour)
        )

        subset = End.loc[mask]

        if subset.empty:
            return {"error": "Date not found"}

        forecast = subset["Forecasted"].sum()

        shap_values = (
            subset
            .drop(columns=["Forecasted", "Date"])
            .mean()
            .to_dict()
        )

        return {
            "aggregation": aggregation,
            "forecast": float(forecast),
            "Shapley": shap_values
        }

    elif aggregation == "daily":
        mask = End["Date"].dt.date == date.date()
        subset = End.loc[mask]

        if subset.empty:
            return {"error": "Date not found"}

        forecast = subset["Forecasted"].sum()

        shap_values = (
            subset
            .drop(columns=["Forecasted", "Date"])
            .mean()
            .to_dict()
        )

        return {
            "aggregation": aggregation,
            "forecast": float(forecast),
            "Shapley": shap_values
        }

# ===============================
# UI
# ===============================

from datetime import datetime,timedelta


selected_date = st.date_input(
    "Select date",
    value=datetime.today(),            # Default = today
    min_value=datetime.today(),    # Minimum selectable date
    max_value=datetime.today()+timedelta(days=3)   # Maximum selectable date
)

selected_time = st.time_input(
    "Select time",
    value=datetime(2025,1,1,0,0,0).time()  # Default current time
)

selected_date=selected_date+timedelta(hours=selected_time.hour)+timedelta(minutes=selected_time.minute)

aggregation=st.sidebar.selectbox("Aggregation level",options=['daily','hourly','quarter_hour'])

n_to_choice = st.sidebar.number_input("Enter number of feature to select:", min_value=0, max_value=End.shape[1]-2, step=1)


if st.button("Run analysis"):


    result = shap_return(str(selected_date),aggregation)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    forecast = result["forecast"]
    shap_dict = result["Shapley"]
    # aggregation = result["aggregation"]
    

    shap_series = pd.Series(shap_dict)

    top_positive = shap_series.sort_values(ascending=False).head(n_to_choice)
    top_negative = shap_series.sort_values().head(n_to_choice)

    # ===============================
    # METRIC
    # ===============================

    st.subheader(f"Aggregation level: {aggregation}")
    # row = End.loc[End["Date"] == selected_date,'Forecasted']
    col1,col2=st.columns(2)
    # with col1:
    st.metric("Total Forecast", f"{forecast:,.2f}")
    # with col2:
    # st.line_chart(row)

    # ===============================
    # SHAP VISUALIZATION
    # ===============================

    with col1:
        st.subheader("Top 5 Positive Impact")
        st.bar_chart(top_positive)

    with col2:
        st.subheader("Top 5 Negative Impact")
        st.bar_chart(top_negative)

    # ===============================
    # LLM INTERPRETATION
    # ===============================

    explanation_prompt = f"""
    Forecast totale: {forecast}

    Top 5 feature positive:
    {top_positive.to_dict()}

    Top 5 feature negative:
    {top_negative.to_dict()}

    Scrivi una spiegazione manageriale chiara e sintetica (massimo 8 righe).
    """
    with st.spinner("Generating interpretation..."):
        
        st.subheader("LLM Interpretation")
    
        # Create empty placeholder for streaming text
        explanation_placeholder = st.empty()
        full_response = ""
    
        # Streaming completion
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un analista senior esperto di forecast energetici."},
                {"role": "user", "content": explanation_prompt}
            ],
            temperature=0.3,
            stream=True  # Enable streaming
        )
    
        # Iterate over streamed chunks
        for chunk in stream:
            # Extract content delta
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                
                # Update UI progressively
                with st.expander("📂 LLM Interpretation"):
                    explanation_placeholder.markdown(full_response)    
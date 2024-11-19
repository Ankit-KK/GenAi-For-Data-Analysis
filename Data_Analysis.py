import streamlit as st
import pandas as pd
import re
from openai import OpenAI

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["API_KEY"]
    )

# Convert dataset to a formatted string
def dataset_to_string(df):
    try:
        data_sample = df.head().to_string()
        data_info = df.describe(include='all').to_string()
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
        return ""
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

# Generate an enhanced EDA prompt
def create_eda_prompt(data_str):
    return f"""
    **Role**: You are an advanced data analyst and visualization expert.

    I have provided you with a dataset for performing a detailed exploratory data analysis (EDA). Your task is to identify trends, relationships, and anomalies in the dataset using statistical and visualization techniques.

    ### Dataset Overview:
    - **Data Sample:**
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```

    - **Data Description:**
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    ### Tasks:

    **1. Data Inspection:**
       - Summarize dataset structure (e.g., shape, columns, data types).
       - Identify missing values and outliers, suggesting appropriate strategies to handle them.

    **2. Descriptive Statistics:**
       - Compute key statistics (mean, median, mode, standard deviation, skewness, kurtosis).
       - Highlight any noteworthy trends or anomalies.

    **3. Visual Exploration:**
       - Plot histograms, box plots, and density plots for numerical features.
       - Use bar plots or count plots for categorical variables.
       - Generate scatter plots, pair plots, and correlation heatmaps to explore relationships.

    **4. Advanced Visualizations:**
       - Use violin plots and swarm plots to visualize distributions.
       - Apply clustering techniques (e.g., K-Means or DBSCAN) for grouping insights.
       - Perform Principal Component Analysis (PCA) for dimensionality reduction and visualize in 2D/3D.

    **5. Feature Relationships:**
       - Analyze relationships between features and a target variable (if applicable).
       - Use grouped bar charts, trendlines, or advanced statistical tests to uncover patterns.

    **6. Recommendations and Next Steps:**
       - Summarize insights, patterns, and anomalies observed in the data.
       - Provide actionable recommendations, including ideas for feature engineering and preprocessing steps.

    ### Output Requirements:
    - Python code for each step with detailed comments.
    - Use libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
    - Provide clean and modular code that is ready for execution.
    - Include explanations and visualizations in the output to ensure interpretability.
    """

# Preprocess the generated code
def preprocess_generated_code(code):
    code = re.sub(r'```python|```', '', code)
    return code.strip()

import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import json
from datetime import datetime

# ... existing OpenAI client and dataset functions remain the same ...

def load_feedback(file_path='feedback.txt', max_entries=5):
    try:
        with open(file_path, 'r') as f:
            feedbacks = [json.loads(line) for line in f]
            return feedbacks[-max_entries:]  # Return last 5 entries
    except FileNotFoundError:
        return []

def save_feedback(rating, text, email, file_path='feedback.txt'):
    feedback = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'rating': rating,
        'text': text,
        'email': email
    }
    
    with open(file_path, 'a') as f:
        f.write(json.dumps(feedback) + '\n')

def main():
    st.title("Advanced Exploratory Data Analysis with Llama")

    client = get_openai_client()

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if st.button("Generate EDA Code"):
            data_str = dataset_to_string(df)
            if not data_str:
                return

            eda_prompt = create_eda_prompt(data_str)

            try:
                with st.spinner("Generating EDA code..."):
                    completion = client.chat.completions.create(
                        model="meta/llama-3.2-3b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.5,
                        top_p=0.9,
                        max_tokens=2048,
                        stream=True
                    )

                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            generated_code += chunk.choices[0].delta.content

                processed_code = preprocess_generated_code(generated_code)
                st.subheader("Generated EDA Code:")
                st.code(processed_code)

                file_path = "eda_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)

                with open(file_path, "r") as f:
                    st.download_button("Download Generated Code", f, file_name="eda_generated.py", mime="text/plain")

            except Exception as e:
                st.error(f"Error generating EDA code: {e}")



    # Add feedback form
    st.markdown("---")
    st.subheader("Feedback")
    
    with st.form(key="feedback_form"):
        feedback_rating = st.slider("How would you rate this tool?", 1, 5, 3)
        feedback_text = st.text_area("Please share your feedback or suggestions:")
        feedback_email = st.text_input("Email (optional):", "")
        submit_button = st.form_submit_button(label="Submit Feedback")
        
        if submit_button:
            save_feedback(feedback_rating, feedback_text, feedback_email)
            st.success("Thank you for your feedback!")

    # Display recent feedbacks only if they exist
    recent_feedbacks = load_feedback()
    if recent_feedbacks:
        st.markdown("---")
        st.subheader("Recent Feedbacks")
        
        for feedback in recent_feedbacks:
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"Rating: {feedback['rating']}/5")
                with col2:
                    st.write(f"Comment: {feedback['text']}")
                st.write(f"Date: {feedback['timestamp']}")
                st.markdown("---")
    else:
        st.write("No feedbacks yet!")

if __name__ == "__main__":
    main()

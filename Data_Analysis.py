import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import os

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

# Main Streamlit app function
def main():
    st.title("ExploraGen: Advanced Exploratory Data Analysis with Llama")

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
                        model="meta/llama-3.2-1b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=1024,
                        stream=True
                    )

                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            generated_code += chunk.choices[0].delta.content

                # Process and display the generated code
                processed_code = preprocess_generated_code(generated_code)
                st.subheader("Generated EDA Code:")
                st.code(processed_code)

                # Provide download option
                file_path = "eda_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)

                with open(file_path, "r") as f:
                    st.download_button("Download Generated Code", f, file_name="eda_generated.py", mime="text/plain")

            except Exception as e:
                st.error(f"Error generating EDA code: {e}")

    # Feedback Section using Google Form
    st.sidebar.subheader("We Value Your Feedback")
    st.sidebar.markdown("""
    <a href="https://forms.gle/rTrFC4rwqfJ9B6mE9" target="_blank">
        <button style="
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 14px; 
            margin: 4px 2px; 
            cursor: pointer;
            border: none;
            border-radius: 8px;
        ">
            Open Feedback Form
        </button>
    </a>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

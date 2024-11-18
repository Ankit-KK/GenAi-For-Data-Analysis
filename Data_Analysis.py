import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import base64

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["Api_key"]
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

# Generate EDA prompt with in-depth visualizations and analysis
def create_eda_prompt(data_str):
    return f"""
    You are a data analyst specializing in advanced exploratory data analysis (EDA). Analyze the dataset provided below and perform the following tasks:

    **Dataset Overview:**
    - Data Sample:
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```

    - Data Description:
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    **Tasks:**

    1. **Dataset Inspection:**
       - Inspect the dataset structure (columns, data types, and dimensions).
       - Identify missing values and outliers. Provide recommendations for handling them.

    2. **Descriptive Statistics:**
       - Compute summary statistics (mean, median, mode, variance, standard deviation, min, max).
       - Provide insights into the data distributions (e.g., skewness, kurtosis).

    3. **In-depth Visualizations:**
       - Plot histograms and density plots for numerical features to understand their distributions.
       - Create scatter plots and pair plots to explore relationships between numerical variables.
       - Generate box plots to identify outliers and interquartile ranges.
       - Use bar plots or count plots to visualize the distribution of categorical variables.
       - Provide advanced visualizations, such as violin plots or swarm plots, where applicable.

    4. **Correlation Analysis:**
       - Compute and visualize a correlation matrix with a heatmap.
       - Identify pairs of features with high correlation. Suggest methods to handle multicollinearity.

    5. **Feature Relationships:**
       - Analyze the relationship between features and a target variable (if specified).
       - Use grouped bar charts, scatter plots with trendlines, or box plots to highlight significant relationships.

    6. **Advanced Techniques:**
       - Perform Principal Component Analysis (PCA) to reduce dimensionality and visualize the data in 2D/3D.
       - Use clustering (e.g., K-Means or Hierarchical Clustering) to identify groups or patterns in the data.

    7. **Insights and Recommendations:**
       - Summarize findings, including trends, anomalies, and patterns.
       - Provide actionable recommendations, such as feature engineering ideas or additional analyses.
       - Suggest data cleaning or preprocessing steps required for further modeling or analysis.

    **Output Format:**
    - Provide Python code for each task with detailed comments.
    - Include visualizations using matplotlib, seaborn, or other libraries.
    - Use clear print statements to explain the insights from each step.
    """

# Preprocess generated code
def preprocess_generated_code(code):
    code = re.sub(r'```python|```', '', code)
    return code.strip()

# Main Streamlit app function
def main():
    st.title("In-depth Exploratory Data Analysis with Llama")

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
                        model="meta/llama-3.1-405b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.5,
                        top_p=0.7,
                        max_tokens=2048,
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

                with open(file_path, 'r') as f:
                    st.download_button("Download Generated Code", f, file_name=file_path, mime="text/plain")

            except Exception as e:
                st.error(f"Error generating EDA code: {e}")

if __name__ == "__main__":
    main()

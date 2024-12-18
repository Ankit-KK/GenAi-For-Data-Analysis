import streamlit as st
import pandas as pd
import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Initialize NVIDIA AI client
def get_nvidia_client(api_key):
    return ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
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
    - Include explanations and visualizations in the output to ensure interpretability."""

# Preprocess the generated code
def preprocess_generated_code(code):
    code = re.sub(r'```python|```', '', code)
    return code.strip()

# Main Streamlit app function
def main():
    st.title("ExploraGen: Advanced Exploratory Data Analysis with Llama 3.2")

    st.sidebar.subheader("I appreciate your feedback.")
    st.sidebar.markdown(
        """
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
                Submit Feedback
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("Get Your Free 1000 Credits here: [NVIDIA AI Free Credits](https://build.nvidia.com/meta/llama-3_1-405b-instruct?snippet_tab=LangChain)")
    api_key = st.text_input("Enter your NVIDIA AI API Key:", type="password")

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    client = get_nvidia_client(api_key)

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
                    generated_code = ""
                    for chunk in client.stream([{"role": "user", "content": eda_prompt}]):
                        generated_code += chunk.content

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

if __name__ == "__main__":
    main()

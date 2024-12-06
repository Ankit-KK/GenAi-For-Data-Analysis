import streamlit as st
import pandas as pd
import re
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import traceback



# Initialize NVIDIA AI client
@st.cache_resource
def get_nvidia_client():

    st.markdown("Get Your Free 1000 Credits here: [NVIDIA AI Free Credits](https://build.nvidia.com/meta/llama-3_1-405b-instruct?snippet_tab=LangChain)")
    
    api_key = st.text_input("Enter your NVIDIA AI API Key:", type="password")
    
    if api_key:
        st.success("API Key successfully entered!")
    else:
        st.info("Please enter your API key to proceed.")
    return ChatNVIDIA(
        model="meta/llama-3.1-405b-instruct",  # Updated NVIDIA model
        api_key= api_key,
        temperature=0.5,
        top_p=0.9,
        max_tokens=2048,
    )

# Convert dataset to a formatted string
def dataset_to_string(df):
    try:
        data_sample = df.head().to_string()
        data_info = df.describe(include="all").to_string()
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
        return ""
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

# Generate an enhanced EDA prompt
def create_eda_prompt(data_str):
    return """

**Your Role:**

You are a seasoned data scientist, adept at extracting valuable insights from complex datasets. Your task is to conduct a comprehensive exploratory data analysis (EDA) on a dataset provided below. Through statistical analysis and compelling visualizations, you will uncover hidden patterns, identify anomalies, and derive actionable insights.

**Dataset Overview:**

* **Data Sample:** 
  ```
  {data_str.split('Data Description:')[0].strip()}
  ```

* **Data Description:**
  ```
  {data_str.split('Data Description:')[1].strip()}
  ```

**Your Mission:**

1. **Data Deep Dive:**
   * **Unmask the Data:** Delve into the dataset's structure, shape, and data types.
   * **Missing Pieces and Outliers:** Identify and handle missing values and outliers, justifying your approach.

2. **Statistical Portrait:**
   * **Key Metrics:** Calculate essential statistics (mean, median, mode, standard deviation, skewness, kurtosis) to understand data distribution.
   * **Spotting the Unusual:** Highlight any intriguing trends or anomalies that warrant further investigation.

3. **Visual Storytelling:**
   * **Data Visualization:** Employ histograms, box plots, and density plots to visualize numerical variables.
   * **Categorical Insights:** Utilize bar plots or count plots to explore categorical data.
   * **Relationship Revelations:** Generate scatter plots, pair plots, and correlation heatmaps to uncover relationships between variables.

4. **Advanced Visual Exploration:**
   * **Deeper Insights:** Employ violin plots and swarm plots to visualize distributions in greater detail.
   * **Clustering for Clarity:** Apply clustering techniques (K-Means, DBSCAN) to group similar data points and identify underlying patterns.
   * **Dimensionality Reduction:** Utilize Principal Component Analysis (PCA) to reduce dimensionality and visualize data in 2D or 3D space.

5. **Feature Relationships and Target Variable:**
   * **Feature Impact:** Analyze the relationship between features and a target variable (if applicable).
   * **Visualizing Trends:** Employ grouped bar charts, trendlines, or advanced statistical tests to uncover hidden patterns.

6. **Actionable Insights and Future Directions:**
   * **Summarize Key Findings:** Concisely present the major insights, patterns, and anomalies discovered during the EDA.
   * **Data-Driven Recommendations:** Provide actionable recommendations, including potential feature engineering techniques and preprocessing steps to enhance future modeling efforts.

**Deliverables:**

* **Python Code:** Present clean, well-commented Python code using libraries like pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
* **Clear and Concise Explanations:** Accompany your code with clear explanations and visualizations to ensure easy interpretation of results. 
* **Reproducible Workflow:** Ensure your code is modular and ready for execution, facilitating reproducibility and collaboration.

"""


# Preprocess the generated code
def preprocess_generated_code(code):
    code = re.sub(r"```python|```", "", code)
    return code.strip()

# Main Streamlit app function
def main():
    st.title("ExploraGen: Advanced Exploratory Data Analysis with NVIDIA AI")

    # Feedback Section using Google Form
    st.sidebar.subheader("I appreciate your feedback.")
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
            Submit Feedback
        </button>
    </a>
    """, unsafe_allow_html=True)

    client = get_nvidia_client()

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
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()

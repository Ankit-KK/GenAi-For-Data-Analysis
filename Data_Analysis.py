import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import traceback

# Initialize OpenAI client with your NVIDIA API base URL and API key
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["api_key"]  # Store your API key in Streamlit secrets
    )

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return data_sample, data_info

def create_eda_prompt(data_sample, data_info):
    """Create a custom EDA prompt for the language model."""
    return f"""
    **Role**: You are an expert data analyst.

    **Context**: I have provided you with a dataset containing various features. Your task is to perform a comprehensive exploratory data analysis (EDA) to uncover insights, patterns, and potential issues in the data. The dataset includes a mix of numerical and categorical features, and it is crucial to explore these thoroughly to inform further analysis or decision-making.

    **Dataset Overview**:
    - **Data Sample**:
      ```
      {data_sample}
      ```

    - **Data Description**:
      ```
      {data_info}
      ```

    **Tasks**:

    1. **Data Overview**:
       - Print "Executing Task 1: Data Overview"
       - Inspect the first few rows of the dataset to understand its structure.
       - Determine the data types of each column (numerical, categorical, etc.).
       - Check for missing values and describe the proportion of missing data in each column.

    2. **Descriptive Statistics**:
       - Print "Executing Task 2: Descriptive Statistics"
       - Calculate summary statistics (mean, median, mode, standard deviation, variance, minimum, maximum) for each numerical column.
       - Provide insights on the distribution of these numerical features (e.g., skewness, kurtosis).

    3. **Data Visualization**:
       - Print "Executing Task 3: Data Visualization"
       - Plot histograms and density plots for each numerical column to visualize distributions.
       - Create scatter plots to examine relationships between key numerical variables (e.g., feature vs. target variable).
       - Use box plots to identify outliers and understand the spread of the data.

    4. **Categorical Data Analysis**:
       - Print "Executing Task 4: Categorical Data Analysis"
       - Summarize the frequency of each category within categorical columns.
       - Use bar plots or count plots to visualize the distribution of categorical variables.
       - Analyze the relationship between categorical variables and the target variable (if applicable), using grouped bar charts or other appropriate visualizations.

    5. **Correlation Analysis**:
       - Print "Executing Task 5: Correlation Analysis"
       - Compute the correlation matrix for numerical features.
       - Visualize the correlation matrix using a heatmap and identify pairs of highly correlated features.
       - Discuss potential implications of multicollinearity and suggest strategies for dealing with it.

    6. **Advanced Analysis**:
       - Print "Executing Task 6: Advanced Analysis"
        - **Handle Missing Values:**
            - Check for missing values in the dataset.
            - If missing values are present:
                - Choose an appropriate strategy (e.g., imputation, dropping rows/columns) based on the type and extent of missingness.
                - Explain the rationale behind the chosen strategy and its potential impact on the analysis.
                - Implement the chosen strategy to handle missing values.
        - Perform clustering (e.g., K-means) or dimensionality reduction (e.g., PCA) on the preprocessed data to uncover patterns or groupings in the data.
       - Identify any anomalies or unusual patterns that might warrant further investigation.

    7. **Insights and Recommendations**:
       - Print "Executing Task 7: Insights and Recommendations"
       - Summarize the key findings from the EDA, highlighting significant patterns, trends, or anomalies.
       - Provide actionable insights based on the analysis, such as data cleaning steps, feature engineering ideas, or further analyses that could be conducted.
       - Suggest potential next steps, including any additional data that may be required or further analyses that could enhance understanding.

    **Instructions for Model**:
    - Provide Python code snippets for each task, ensuring that the code is efficient, well-commented, and easy to understand.
    - Include print statements before each task to indicate which task is being executed.
    - Execute the code snippets where necessary to validate the findings and ensure there are no errors.
    - If any assumptions are made during the analysis, clearly state them and explain their rationale.

    **Output**:
    - The analysis should be comprehensive and thorough, providing clear and actionable insights based on the data.
    - Include any visualizations as part of the output to support the findings and provide a clear understanding of the data.
    """

def main():
    st.title("Comprehensive Exploratory Data Analysis")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if st.button("Generate EDA"):
            data_sample, data_info = dataset_to_string(df)
            eda_prompt = create_eda_prompt(data_sample, data_info)

            client = get_openai_client()

            with st.spinner("Generating EDA code..."):
                try:
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=2048,
                        stream=True
                    )
                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            generated_code += chunk.choices[0].delta.content

                    # Preprocess the generated code
                    processed_code = generated_code.replace("'''", '"""').replace("''", '"')
                    if "import matplotlib.pyplot as plt" not in processed_code:
                        processed_code = "import matplotlib.pyplot as plt\n" + processed_code
                    if "import seaborn as sns" not in processed_code:
                        processed_code = "import seaborn as sns\n" + processed_code

                    st.subheader("Generated Code:")
                    st.code(processed_code)

                    # Save to Python file
                    file_path = "EDA_generated.py"
                    with open(file_path, "w") as f:
                        f.write(processed_code)
                    st.success(f"Generated code saved to '{file_path}'")

                    # Warning message about potential code adjustments
                    st.warning("The generated code might contain minor errors or require slight adjustments.")

                except Exception as e:
                    st.error("Error generating code.")
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
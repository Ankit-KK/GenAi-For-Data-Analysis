import streamlit as st
import pandas as pd
import re
import base64
from openai import OpenAI

# Initialize OpenAI client with the NVIDIA API
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["API_KEY"]  # Securely fetch the API key from Streamlit secrets
    )

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all', datetime_is_numeric=True).to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_eda_prompt(data_str):
    """Create a custom EDA prompt for the language model."""
    return f"""
    You are a data analysis expert.

    I have provided a dataset that contains both numerical and categorical features. Your task is to perform a thorough exploratory data analysis (EDA) to uncover insights, patterns, and any potential issues in the data.

    **Dataset Overview**:
    - **Data Sample**:
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```
    - **Data Description**:
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    **Tasks**:
    1. Overview of data structure, missing values, and column data types.
    2. Descriptive statistics for numerical columns (mean, median, standard deviation, etc.).
    3. Visualization of numerical and categorical distributions (histograms, bar plots).
    4. Correlation analysis with a heatmap.
    5. Suggestions for handling missing values and potential data issues.

    Provide Python code for these tasks, ensuring it is well-commented and ready for execution.
    """

def preprocess_generated_code(code):
    """Clean up and standardize generated Python code."""
    code = re.sub(r'```python|```', '', code)  # Remove markdown formatting
    return code.strip()

def main():
    st.title("Exploratory Data Analysis with NVIDIA Meta LLaMA 3.2")
    st.write("Upload a dataset and let the model generate Python code for comprehensive EDA.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if st.button("Generate EDA Code"):
            data_str = dataset_to_string(df)
            eda_prompt = create_eda_prompt(data_str)
            client = get_openai_client()

            try:
                with st.spinner("Generating EDA code with Meta LLaMA 3.2..."):
                    completion = client.chat.completions.create(
                        model="meta/llama-3.2-405b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=2048,
                        stream=True
                    )

                    # Collect the generated code
                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            generated_code += chunk.choices[0].delta.content

                # Preprocess and display the generated code
                processed_code = preprocess_generated_code(generated_code)
                st.subheader("Generated EDA Code:")
                st.code(processed_code, language="python")

                # Save to file
                file_path = "eda_generated_llama32.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)
                st.success("Code generation complete! You can download the Python file below.")
                st.download_button("Download EDA Code", processed_code, file_name=file_path, mime="text/plain")

            except Exception as e:
                st.error("An error occurred while generating the EDA code.")
                st.exception(e)

if __name__ == "__main__":
    main()

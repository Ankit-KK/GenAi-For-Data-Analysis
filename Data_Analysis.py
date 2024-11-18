import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import traceback
from io import StringIO
import re
import base64
import os

# Initialize OpenAI client with a custom API key from Streamlit secrets
@st.cache_resource
def get_openai_client():
    api_key = st.secrets["api_key"]
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_eda_prompt(data_str):
    """Create a custom EDA prompt for the language model."""
    return f"""
    **Role**: You are an expert data analyst.

    **Context**: I have provided you with a dataset containing various features. Your task is to perform a comprehensive exploratory data analysis (EDA) to uncover insights, patterns, and potential issues in the data. The dataset includes a mix of numerical and categorical features, and it is crucial to explore these thoroughly to inform further analysis or decision-making.

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
    ... (remaining prompt tasks)
    """

def preprocess_generated_code(code):
    # Remove any markdown code block indicators
    code = re.sub(r'```python|```', '', code)
    
    # Remove any explanatory text before the actual code
    code = re.sub(r'^.*?import', 'import', code, flags=re.DOTALL)
    
    # Replace triple quotes with double quotes
    code = code.replace("'''", '"""')
    
    # Ensure necessary imports are present
    if "import matplotlib.pyplot as plt" not in code:
        code = "import matplotlib.pyplot as plt\n" + code
    if "import seaborn as sns" not in code:
        code = "import seaborn as sns\n" + code
    
    return code.strip()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("ExploraGen")

    # Load API key from Streamlit secrets
    api_key = st.secrets["API_KEY"]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if st.button("Generate EDA Code"):
            data_str = dataset_to_string(df)
            eda_prompt = create_eda_prompt(data_str)

            client = get_openai_client()

            try:
                with st.spinner("Generating EDA code..."):
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=[{"role": "user", "content": eda_prompt}],
                        temperature=0.5,
                        top_p=0.7,
                        max_tokens=2048,
                        stream=True
                    )

                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            generated_code += chunk.choices[0].delta.content

                # Preprocess the generated code
                processed_code = preprocess_generated_code(generated_code)

                st.subheader("Generated Code:")
                st.code(processed_code)

                # Save to Python file
                file_path = "eda_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)
                st.success(f"Generated code saved to '{file_path}'")

                # Add download button for the generated Python file
                with open(file_path, 'r') as f:
                    st.download_button('Download EDA Code', f, file_name=file_path, mime='text/plain')

                # Warning message about potential code adjustments
                st.warning("The generated code might contain minor errors or require slight adjustments.")

            except Exception as e:
                st.error("An error occurred while generating the EDA code.")
                st.exception(e)

if __name__ == "__main__":
    main()

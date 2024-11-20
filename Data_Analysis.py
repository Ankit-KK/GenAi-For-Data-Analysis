import streamlit as st
import pandas as pd
import re
from openai import OpenAI

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key= st.secrets["API_KEY"]
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
    (rest of the prompt remains unchanged)
    """

# Preprocess the generated code
def preprocess_generated_code(code):
    code = re.sub(r'```python|```', '', code)
    return code.strip()

# Main Streamlit app function
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

if __name__ == "__main__":
    main()

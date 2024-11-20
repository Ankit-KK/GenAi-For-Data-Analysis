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
        api_key= st.secrets["API_KEY"]
    )

# Feedback file path in the app's directory
FEEDBACK_FILE = "feedback.csv"

# Initialize feedback file if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.DataFrame(columns=["Email_id", "Feedback"])
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

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

# Save feedback to the local feedback.csv file
def save_feedback(email, feedback):
    try:
        # Handle empty email gracefully
        email = email.strip() if email else ""  # Ensure blank emails are stored as empty strings
        new_feedback = pd.DataFrame({"Email_id": [email], "Feedback": [feedback]})
        
        # Load existing feedback data
        if os.path.exists(FEEDBACK_FILE):
            existing_feedback = pd.read_csv(FEEDBACK_FILE)
        else:
            existing_feedback = pd.DataFrame(columns=["Email_id", "Feedback"])
        
        # Append the new feedback
        updated_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        updated_feedback.to_csv(FEEDBACK_FILE, index=False)
        
        # Log success
        st.success("Feedback saved successfully!")
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

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

    # Feedback Section
    st.sidebar.subheader("We Value Your Feedback")
    with st.sidebar.form(key="feedback_form"):
        email = st.text_input("Email (optional)")
        feedback = st.text_area("Your Feedback")
        st.caption("Email ID will be hidden.")
        if st.form_submit_button("Submit Feedback"):
            if feedback.strip():  # Ensure feedback is not empty
                save_feedback(email, feedback)
            else:
                st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import requests
import base64

# GitHub Configuration
GITHUB_TOKEN = "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"
REPO_OWNER = "Ankit-KK"
REPO_NAME = "GenAi-For-Data-Analysis"
FEEDBACK_FILE_PATH = "feedback.csv"

# GitHub API Headers
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Get file content from GitHub
def get_github_file():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FEEDBACK_FILE_PATH}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        # Return an empty DataFrame if the file does not exist
        return {"content": base64.b64encode(pd.DataFrame(columns=["Email_id", "Feedback"]).to_csv(index=False).encode()).decode(), "sha": None}
    else:
        st.error(f"Error fetching file from GitHub: {response.json()}")
        return None

# Update file on GitHub
def update_github_file(new_content, sha):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FEEDBACK_FILE_PATH}"
    data = {
        "message": "Update feedback file",
        "content": base64.b64encode(new_content.encode()).decode(),
        "sha": sha,
    }
    response = requests.put(url, headers=HEADERS, json=data)
    if response.status_code == 200 or response.status_code == 201:
        st.success("Feedback saved successfully!")
    else:
        st.error(f"Error saving feedback to GitHub: {response.json()}")

# Save feedback to GitHub
def save_feedback_to_github(email, feedback):
    file_data = get_github_file()
    if file_data:
        # Decode existing file content
        existing_content = base64.b64decode(file_data["content"]).decode()
        feedback_df = pd.read_csv(pd.compat.StringIO(existing_content))

        # Append new feedback
        new_feedback = pd.DataFrame({"Email_id": [email], "Feedback": [feedback]})
        updated_feedback = pd.concat([feedback_df, new_feedback], ignore_index=True)

        # Save updated content back to GitHub
        update_github_file(updated_feedback.to_csv(index=False), file_data.get("sha"))

# Streamlit app
def main():
    st.title("Feedback with GitHub Integration")

    # Feedback Section
    st.sidebar.subheader("We Value Your Feedback")
    with st.sidebar.form(key="feedback_form"):
        email = st.text_input("Email (optional)")
        feedback = st.text_area("Your Feedback")
        st.caption("Email ID will be hidden.")
        if st.form_submit_button("Submit Feedback"):
            if feedback.strip():  # Ensure feedback is not empty
                save_feedback_to_github(email.strip() if email else "", feedback)
            else:
                st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()

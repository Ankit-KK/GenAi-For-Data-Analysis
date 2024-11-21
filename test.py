import streamlit as st

def feedback_page():
    st.title("Provide Feedback")
    
    st.markdown("""
    ## We Value Your Input!

    Please take a moment to share your feedback by clicking the button below:

    ### Feedback Form
    """)

    # Create a prominent button to open the Google Form
    st.markdown("""
    <a href="https://forms.gle/gExh43NfF3oNETRK6" target="_blank">
        <button style="
            background-color: #4CAF50; 
            color: white; 
            padding: 15px 32px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer;
            border: none;
            border-radius: 8px;
        ">
            Open Feedback Form
        </button>
    </a>
    """, unsafe_allow_html=True)

    # Additional context
    st.markdown("""
    ### Why Your Feedback Matters
    - Help us improve the user experience
    - Share your suggestions and insights
    - Contribute to the future development of our tool

    Your feedback is anonymous and greatly appreciated!
    """)

# In your main Streamlit app, you can integrate this as a page or section
def main():
    # Your existing code...
    
    # Add a navigation option
    page = st.sidebar.radio("Navigate", ["Main App", "Feedback"])
    
    if page == "Feedback":
        feedback_page()
    else:
        # Your existing main app content
        pass

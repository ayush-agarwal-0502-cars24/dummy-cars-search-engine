# app.py

import streamlit as st
from dotenv import load_dotenv
from search_utils import (
    load_vector_db,
    setup_agent,
    get_ai_summary  # <- new helper function
)

# Load environment variables
load_dotenv()

# Initialize vector DB and agent once
db = load_vector_db()
agent = setup_agent()

# Streamlit UI
st.set_page_config(page_title="Car Finder AI", layout="wide")
st.title("🚗 AI Car Search Assistant")

# Search box
query = st.text_input("Enter your car search query:")

if query:
    with st.spinner("🔍 Searching..."):
        docs = db.similarity_search(query, k=5)

        st.subheader("## 📄 Top Matching Cars")

        for i, doc in enumerate(docs):
            car_info = doc.page_content
            ai_summary = get_ai_summary(agent, car_info, query)

            # Show raw car info + detailed AI insight
            st.markdown(
                f"""
                <div style="border: 1px solid #ccc; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                    <strong>Match #{i+1}</strong><br><br>
                    <strong>🔍 Car Info:</strong><br>{car_info}<br><br>
                    <strong>🧠 AI Insight:</strong><br>{ai_summary}
                </div>
                """,
                unsafe_allow_html=True
            )


# Follow-up query
follow_up = st.text_input("Have a follow-up question? (e.g., ‘Can you show me electric options instead?’)")

if follow_up:
    with st.spinner("🤖 Thinking with memory..."):
        follow_up_response = agent.run(follow_up)
        st.subheader("🧠 AI Follow-up Response")
        st.write(follow_up_response)
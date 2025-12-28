import streamlit as st
import random
import time
from utils.utilities import predict, visualize_attributions  # Placeholder for your AI detection logic
# 1. Page Configuration for a cleaner look
st.set_page_config(
    page_title="AIDataHunter",
    layout="centered"
)

# 2. Custom CSS to mimic the clean Gemini look
# We hide the default main menu and footer for a more "app-like" feel
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    /* Style the popover button to look more like a simple icon */
    [data-testid="stPopover"] > div > button {
        border: none;
        background: transparent;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Header
# st.title("AI Data Hunter")
# st.caption("Paste text to detect if it's AI-generated.")

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display text content
        if message.get("content"):
            st.markdown(message["content"])
        # Display image if present
        if message.get("image"):
            st.image(message["image"], width=300)

# # 6. Input Area (The "Gemini" Style Bottom Bar)
# # We use a container to group the input and the upload button visually
# with st.container():
#     # A. The "+" Button (Popover) for Image Uploads
#     # Placed inside a column or just above chat input (Streamlit constraint: input is always at bottom)
#     # To mimic Gemini closely, we put the upload "menu" right above the chat bar.
#     with st.popover("Add Image", use_container_width=False):
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # B. The Chat Input
prompt = st.chat_input("Enter text to analyze for AI generation...")

# 7. Logic: Handle Submission
if prompt:
    # --- USER STEP ---
    # Save user message to state
    user_msg_data = {"role": "user", "content": prompt}
    # if uploaded_file:
    #     user_msg_data["image"] = uploaded_file
    
    st.session_state.messages.append(user_msg_data)
    
    # Display user message immediately in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
        # if uploaded_file:
        #     st.image(uploaded_file, width=300)

    # --- AI STEP (Placeholder) ---
    # Display a loading spinner to look realistic
    with st.chat_message("assistant"):
        with st.spinner("Analyzing content..."):
            # time.sleep(1.5) # Simulate processing time
            prediction, model, input_ids, attention_mask, tokenizer = predict(prompt)
            # Placeholder Logic for AI Detection
            # In the future, you will replace this with your model inference
            ai_condidence = prediction['confidence'][1]
            human_confidence = prediction['confidence'][0]
            if prediction['prediction'] == 1:
                detection_result = f"**Analysis Result** \n\n This content appears to be **AI-Generated** with **{ai_condidence*100:.2f}%** confidence."
            else:
                detection_result = f"**Analysis Result** \n\n This content appears to be **Human-Written** with **{human_confidence*100:.2f}%** confidence."
            
            st.markdown(detection_result)
            with st.spinner("But why this prediction? Generating attributions..."):
                html_visualization = visualize_attributions(model, input_ids, attention_mask, tokenizer, int(prediction['prediction']))
                with st.expander("Why this prediction?"):
                    st.markdown(f"Words with darker reds influence the model more to predict as {('AI-Generated' if prediction['prediction'] == 1 else 'Human-Written')}")
                    st.components.v1.html(html_visualization, height=300, scrolling=True)
    
    # Save AI response to state
    st.session_state.messages.append({"role": "assistant", "content": detection_result})

    # Optional: Clear the uploaded file after sending (requires a rerun or key reset trick, 
    # but for simple UI we leave it as is or you can add a callback)
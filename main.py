import streamlit as st
from rag_chatbot import initialize_rag_system


def main():
    st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📚")
    st.title("📚 Document Q&A Chatbot")

    # Sidebar for HuggingFace token and file upload
    with st.sidebar:
        st.header("Configuration")
        hf_token = st.text_input("HuggingFace Token", type="password")

        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            st.success("Document uploaded successfully!")

    # Chat interface
    if uploaded_file is not None and hf_token:
        # Initialize RAG system with uploaded document and token
        rag_system = initialize_rag_system("uploaded_document.pdf", hf_token)

        # Chat history management
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Ask a question about your document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get RAG response
            with st.chat_message("assistant"):
                response = rag_system.invoke({"question": prompt})
                assistant_response = response["answer"]
                st.markdown(assistant_response)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
    else:
        if not hf_token:
            st.info("Please enter your HuggingFace token in the sidebar.")
        else:
            st.info("Please upload a PDF document to start chatting.")


if __name__ == "__main__":
    main()

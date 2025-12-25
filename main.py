import streamlit as st
from utils import load_documents_from_folder_incremental, get_session_memory, process_uploaded_files, clear_vectorstore_and_cache, log_retrieved_chunks_for_debugging, analyze_chunk_coverage
from analysis import create_chain_with_memory, enhance_query_for_better_retrieval, validate_chunk_relevance

st.set_page_config(
    page_title="AI Doc Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply elegant dark theme CSS
st.markdown("""
<style>
/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    color: #e8eaed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Sidebar elegant styling */
.stSidebar {
    background: linear-gradient(180deg, #1e2329 0%, #2d3748 100%);
    border-right: 1px solid #3a4553;
}

/* Chat messages with modern styling */
.stChatMessage {
    background: rgba(45, 55, 72, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    color: #e8eaed;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* Elegant buttons */
.stButton > button {
    background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
    color: #e8eaed;
    border: 1px solid #4a5568;
    border-radius: 12px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5a6578 0%, #3d4758 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* Modern input styling - excluding chat input */
.stTextInput > div > div > input {
    background: rgba(45, 55, 72, 0.8) !important;
    color: #e8eaed !important;
    border: 1px solid #4a5568 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border: 1px solid #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    outline: none !important;
}

/* Let chat input use default Streamlit styling */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] *,
div[data-testid="stChatInputTextArea"] {
    /* Remove all custom overrides for chat input */
}

/* Metrics styling */
.metric-container {
    background: rgba(45, 55, 72, 0.4);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #1a1f2e;
}
::-webkit-scrollbar-thumb {
    background: #4a5568;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #5a6578;
}
</style>
""", unsafe_allow_html=True)

# Initialize session memory
if "memory" not in st.session_state:
    st.session_state.memory = get_session_memory()

# Load documents and vectorstore
if "vectorstore" not in st.session_state:
    with st.spinner("Loading documents from backend folder..."):
        try:
            new_documents, vectorstore, processing_info = load_documents_from_folder_incremental()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.processing_info = processing_info
                total = processing_info['total_documents']
                new = processing_info['new_documents']
                reused = processing_info['reused_documents']
                # if new > 0:
                #     st.success(f"✅ Processing Complete! Total: {total} docs | New: {new} | Reused: {reused}")
                # else:
                #     st.info(f"✅ Documents Loaded! Total: {total} docs (all from cache)")
            else:
                st.error("No documents found in the backend folder. Please add documents to the 'sop_documents' folder.")
                st.info("📁 Add your SOP documents (PDF, DOCX, CSV, XLSX, TXT) to the 'sop_documents' folder and restart the app.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            st.info("💡 Make sure your documents are in supported formats and not corrupted.")
            st.stop()

# Initialize conversation chain
if "conversation_chain" not in st.session_state:
    try:
        st.session_state.conversation_chain = create_chain_with_memory(
            st.session_state.vectorstore, 
            st.session_state.memory
        )
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        st.stop()

# Clean Professional Sidebar
with st.sidebar:
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;'>
        <h2 style='
            color: #e8eaed; 
            font-size: 1.6rem; 
            font-weight: 600;
            margin-bottom: 0.8rem;
        '>📁 Document Manager</h2>
        <div style='height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 1px; margin: 0 1.5rem;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Document Upload Section
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h3 style='color: #e8eaed; font-size: 1.1rem; font-weight: 500; margin-bottom: 1rem; text-align: center;'>
            📤 Upload Documents
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, CSV, XLSX",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown("""
        <div style='background: rgba(72, 187, 120, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #48bb78; margin: 1rem 0;'>
            <div style='font-size: 0.9rem; color: #68d391; font-weight: 500;'>
                ✅ Files ready to upload
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Process uploaded files - save to sop_documents and add to vectorstore
                    processed_count, updated_vectorstore = process_uploaded_files(uploaded_files)
                    
                    if processed_count > 0:
                        # Update session state with new vectorstore
                        st.session_state.vectorstore = updated_vectorstore
                        
                        # Recreate conversation chain with updated vectorstore
                        st.session_state.conversation_chain = create_chain_with_memory(
                            st.session_state.vectorstore, 
                            st.session_state.memory
                        )
                        
                        # Update processing info
                        _, _, processing_info = load_documents_from_folder_incremental()
                        st.session_state.processing_info = processing_info
                        
                        st.success(f"✅ Successfully processed {processed_count} documents!")
                        st.success(f"📁 Files saved to sop_documents folder")
                        st.success(f"🧠 Vector database updated with {processed_count} new documents")
                    else:
                        st.warning("⚠️ No documents could be processed. Please check file formats.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")
                    
                st.rerun()
    
    # Divider
    st.markdown("""
    <div style='margin: 2rem 0;'>
        <div style='height: 1px; background: rgba(255, 255, 255, 0.1); margin: 0 1rem;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h3 style='color: #e8eaed; font-size: 1.1rem; font-weight: 500; margin-bottom: 1rem; text-align: center;'>
            ⚡ Quick Actions
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons in a clean layout
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.memory.clear()
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
        st.success("✨ Chat history cleared!")
        st.rerun()
    
    if st.button("🔄 Refresh Documents", use_container_width=True):
        with st.spinner("🔍 Refreshing..."):
            try:
                new_documents, vectorstore, processing_info = load_documents_from_folder_incremental()
                if new_documents:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processing_info = processing_info
                    st.session_state.conversation_chain = create_chain_with_memory(
                        st.session_state.vectorstore, 
                        st.session_state.memory
                    )
                    st.success(f"✅ Found {len(new_documents)} new documents!")
                else:
                    st.info("ℹ️ No new documents found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            st.rerun()
    
    # Danger Zone - Clear Vector Database
    st.markdown("""
    <div style='margin: 1.5rem 0;'>
        <div style='height: 1px; background: rgba(255, 100, 100, 0.3); margin: 0 1rem;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🗑️ Clear Vector Database", expanded=False):
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444; margin: 0.5rem 0;'>
            <div style='font-size: 0.85rem; color: #f87171; line-height: 1.4;'>
                ⚠️ <strong>Warning:</strong> This will permanently delete your vector database and processed files cache. 
                All documents will need to be reprocessed when you restart.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Database & Cache", use_container_width=True, type="secondary"):
            with st.spinner("Clearing vector database..."):
                try:
                    success = clear_vectorstore_and_cache()
                    if success:
                        # Clear session state
                        if "vectorstore" in st.session_state:
                            del st.session_state.vectorstore
                        if "processing_info" in st.session_state:
                            del st.session_state.processing_info
                        if "conversation_chain" in st.session_state:
                            del st.session_state.conversation_chain
                        
                        st.success("✅ Vector database and cache cleared successfully!")
                        st.info("💡 Please restart the app or refresh documents to process new files.")
                    else:
                        st.info("ℹ️ No vector database found to clear.")
                except Exception as e:
                    st.error(f"❌ Error clearing database: {e}")
                
            st.rerun()
    
    # Debug Mode Toggle for Automation Team
    st.markdown("""
    <div style='margin: 1.5rem 0;'>
        <div style='height: 1px; background: rgba(255, 193, 7, 0.3); margin: 0 1rem;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <h3 style='color: #e8eaed; font-size: 1rem; font-weight: 500; text-align: center;'>
            🔧 Debug Mode
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple debug toggle for automation team
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False  # Default off for clean UI
    
    st.session_state.debug_mode = st.checkbox(
        "🔧 Debug Mode - Show Retrieved Chunks", 
        value=st.session_state.debug_mode,
        help="Click to see which chunks are retrieved for each query"
    )
    
    # Strict Mode Toggle
    if "strict_mode" not in st.session_state:
        st.session_state.strict_mode = True  # Default strict for accuracy
    
    st.session_state.strict_mode = st.checkbox(
        "🎯 Strict Relevance Mode", 
        value=st.session_state.strict_mode,
        help="Only answer if retrieved chunks are highly relevant to your query"
    )
    
    if st.session_state.debug_mode:
        st.success("✅ Debug mode ON - Chunks will be displayed")
    else:
        st.info("ℹ️ Debug mode OFF - Clean answers only")
        
    if st.session_state.strict_mode:
        st.success("🎯 Strict mode ON - Only highly relevant answers")
    else:
        st.warning("⚠️ Strict mode OFF - May show less relevant results")

    # Help Section
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.2rem; border-radius: 12px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; font-size: 1rem; font-weight: 600; margin-bottom: 0.8rem;'>💡 How to use</h4>
            <div style='font-size: 0.85rem; color: #a0aec0; line-height: 1.5;'>
                1. Upload your documents above<br>
                2. Click "Process Documents"<br>
                3. Start asking questions in the chat<br>
                4. Get AI-powered answers instantly<br>
                5. Enable "Show Retrieved Chunks" to see debug info
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main area: elegant header
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='
            font-size: 4rem; 
            font-weight: 700;
            color: #e8eaed;
            text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            margin-bottom: 0.5rem;
            white-space: nowrap;
            letter-spacing: -2px;
        '>
            🤖 AI Doc Assistant
        </h1>
        <p style='
            color: #a0aec0; 
            font-size: 1.2rem; 
            margin-top: 0;
            font-weight: 300;
        '>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Elegant chat input with enhanced styling
st.markdown("""
<div style='margin: 2rem 0 1rem 0; text-align: center;'>
    <div style='height: 1px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 0 auto; width: 60%;'></div>
</div>
""", unsafe_allow_html=True)

user_input = st.chat_input("Ask AI Doc Assistant")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing your question..."):
                # Enhance the query for better retrieval
                enhanced_query = enhance_query_for_better_retrieval(user_input)
                
                # Use enhanced query for better results
                response = st.session_state.conversation_chain({"question": enhanced_query})
                
                # Validate if retrieved chunks are actually relevant (if strict mode is enabled)
                if "source_documents" in response and response["source_documents"]:
                    chunks = response["source_documents"]
                    
                    # Use strict mode setting to determine validation
                    if st.session_state.get("strict_mode", True):
                        # Strict mode: validate relevance
                        threshold = 0.25  # Higher threshold for strict mode
                        if not validate_chunk_relevance(user_input, chunks, threshold=threshold):
                            assistant_response = "No related information is present in the document."
                            st.warning("⚠️ The retrieved information doesn't seem relevant to your query. Try rephrasing your question or check if the information exists in your documents.")
                        else:
                            assistant_response = response["answer"]
                    else:
                        # Non-strict mode: trust the LLM's judgment
                        assistant_response = response["answer"]
                        st.info("ℹ️ Showing results in non-strict mode - answer may be less relevant")
                else:
                    assistant_response = "No related information is present in the document."
                
                # Display the main answer
                st.markdown(assistant_response)
                
                # Log chunks for automation team debugging (console + file)
                if "source_documents" in response and response["source_documents"]:
                    chunks = response["source_documents"]
                    
                    # Log to console and file for automation team
                    print(f"\n🔍 AUTOMATION TEAM DEBUG - Query: {user_input}")
                    log_retrieved_chunks_for_debugging(
                        query=user_input,
                        chunks=chunks,
                        enhanced_query=enhanced_query if enhanced_query != user_input else None,
                        log_to_file=st.session_state.get("log_to_file", True)
                    )
                    
                    # Additional analysis for automation team
                    analyze_chunk_coverage(chunks)
                
                # Display retrieved chunks for automation team (only if debug mode is enabled)
                if st.session_state.get("show_chunks", True) and "source_documents" in response and response["source_documents"]:
                    chunks = response["source_documents"]
                    
                    # Add toggle to show/hide chunks
                    with st.expander(f"🔍 **Retrieved Chunks ({len(chunks)} chunks used)**", expanded=False):
                        st.markdown("### 📊 Chunks Retrieved for This Query")
                        
                        for i, chunk in enumerate(chunks, 1):
                            # Create a nice card layout for each chunk
                            st.markdown(f"""
                            <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;'>
                                <div style='color: #667eea; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                                    📄 Chunk {i} - {chunk.metadata.get('source', 'Unknown')}
                                </div>
                                <div style='font-size: 0.8rem; color: #a0aec0; margin-bottom: 0.5rem;'>
                                    📁 Type: {chunk.metadata.get('chunk_type', 'standard')} | 
                                    📋 Index: {chunk.metadata.get('chunk_index', 'N/A')} | 
                                    📏 Size: {len(chunk.page_content)} chars
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show chunk content in a code block for easy reading
                            st.text_area(
                                f"Content of Chunk {i}:",
                                value=chunk.page_content,
                                height=150,
                                key=f"chunk_{i}_{len(st.session_state.chat_history)}",
                                disabled=True
                            )
                            
                            # Add separator between chunks
                            if i < len(chunks):
                                st.markdown("---")
                    
                    # Summary information
                    st.info(f"💡 **Query processed using {len(chunks)} chunks** from your document library")
                
                # Show chunk count even when debug mode is disabled
                elif "source_documents" in response and response["source_documents"]:
                    chunks_count = len(response["source_documents"])
                    st.info(f"💡 Query processed using {chunks_count} chunks (enable debug mode to see details)")
                
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            assistant_response = f"❌ Error: {str(e)}"
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.error("💡 Try rephrasing your question or check if your documents contain relevant information.")

import json
import fitz
import pandas as pd
import os
import hashlib
import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as Doc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferWindowMemory
from docx import Document as DocxDocument

def initialize_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_session_memory():
    # OPTION 1: Full session memory - keeps everything
    from langchain_classic.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    return memory

def get_session_memory_windowed():
    # OPTION 2: Window memory - keeps only last 10 exchanges (your old approach)
    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    return memory

def get_session_memory_smart():
    # OPTION 3: Smart hybrid memory - summarizes old, keeps recent in detail
    from langchain_classic.memory import ConversationSummaryBufferMemory
    from analysis import initialise_llm
    
    memory = ConversationSummaryBufferMemory(
        llm=initialise_llm(),
        max_token_limit=1000,  # Keep recent 1000 tokens in detail
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    return memory

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_files_info(config_path="config.json"):
    info_file = "processed_files_info.json"
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_processed_files_info(files_info, config_path="config.json"):
    info_file = "processed_files_info.json"
    try:
        with open(info_file, 'w') as f:
            json.dump(files_info, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files info: {e}")

def save_vectorstore(vectorstore, config_path="config.json"):
    vectorstore_path = "vectorstore"
    try:
        vectorstore.save_local(vectorstore_path)
        print(f"Vectorstore saved to {vectorstore_path}")
    except Exception as e:
        print(f"Error saving vectorstore: {e}")

def load_vectorstore(config_path="config.json"):
    vectorstore_path = "vectorstore"
    if os.path.exists(vectorstore_path):
        try:
            embedding_model = initialize_embedding_model()
            vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"Vectorstore loaded from {vectorstore_path}")
            return vectorstore
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
    return None

def extract_text_from_file_path(file_path):
    """Extract plain text from a file given its path. Only paragraphs from DOCX, selectable text from PDF, and plain text from TXT, CSV, XLSX."""
    file_type = file_path.split(".")[-1].lower()
    text = ""

    if file_type == "docx":
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX file {file_path}: {e}")

    elif file_type == "pdf":
        try:
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")

    elif file_type in ["csv", "xlsx"]:
        try:
            if file_type == "csv":
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
            text = df.to_string()
        except Exception as e:
            print(f"Error reading {file_type} file {file_path}: {e}")
            if file_type == "csv":
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                    text = df.to_string()
                except:
                    print(f"Could not read CSV file {file_path} with any encoding")

    elif file_type == "txt":
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")

    else:
        print(f"Unsupported file type: {file_type}")

    return text

def process_uploaded_files(uploaded_files, config_path="config.json"):
    """Process uploaded files by saving them to sop_documents folder and adding to vectorstore"""
    config = load_config(config_path)
    documents_folder = config.get("documents_folder", "sop_documents")
    
    # Ensure documents folder exists
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
    
    saved_files = []
    processed_documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to sop_documents folder
            file_path = os.path.join(documents_folder, uploaded_file.name)
            
            # Write the uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            saved_files.append(file_path)
            print(f"✅ Saved: {uploaded_file.name}")
            
            # Extract text from the saved file
            document_text = extract_text_from_file_path(file_path)
            
            if document_text.strip():
                # Create document object
                doc = Doc(
                    page_content=document_text,
                    metadata={
                        "source": uploaded_file.name,
                        "file_path": file_path,
                        "file_type": os.path.splitext(uploaded_file.name)[1].lower(),
                        "folder": ""
                    }
                )
                processed_documents.append(doc)
                print(f"✅ Processed: {uploaded_file.name}")
            else:
                print(f"⚠️ No text extracted from: {uploaded_file.name}")
                
        except Exception as e:
            print(f"❌ Error processing {uploaded_file.name}: {e}")
            continue
    
    # Add documents to vectorstore if any were processed
    if processed_documents:
        print(f"Adding {len(processed_documents)} documents to vectorstore...")
        
        # Use optimized chunking strategy for large documents
        print("🔄 Creating optimized chunks...")
        doc_chunks = create_optimized_chunks_for_large_docs(processed_documents, config_path)
        
        # Load existing vectorstore or create new one
        existing_vectorstore = load_vectorstore(config_path)
        
        if existing_vectorstore:
            # Add to existing vectorstore
            existing_vectorstore.add_documents(doc_chunks)
            final_vectorstore = existing_vectorstore
            print("Added documents to existing vectorstore")
        else:
            # Create new vectorstore
            embedding_model = initialize_embedding_model()
            final_vectorstore = FAISS.from_documents(doc_chunks, embedding_model)
            print("Created new vectorstore")
        
        # Save vectorstore
        save_vectorstore(final_vectorstore, config_path)
        
        # Update processed files info
        processed_files_info = load_processed_files_info(config_path)
        for file_path in saved_files:
            file_hash = get_file_hash(file_path)
            processed_files_info[file_path] = {
                "hash": file_hash,
                "name": os.path.basename(file_path),
                "extension": os.path.splitext(file_path)[1].lower(),
                "folder": ""
            }
        save_processed_files_info(processed_files_info, config_path)
        
        return len(processed_documents), final_vectorstore
    
    return 0, load_vectorstore(config_path)

def load_documents_from_folder_incremental(config_path="config.json"):
    config = load_config(config_path)
    documents_folder = config.get("documents_folder", "sop_documents")
    if not os.path.exists(documents_folder):
        print(f"Documents folder '{documents_folder}' not found. Creating it...")
        os.makedirs(documents_folder)
        print(f"Please add your documents to the '{documents_folder}' folder")
        return [], None, {}
    existing_vectorstore = load_vectorstore(config_path)
    processed_files_info = load_processed_files_info(config_path)
    current_files = {}
    supported_extensions = ['.pdf', '.docx', '.csv', '.xlsx', '.txt']
    for root, dirs, files in os.walk(documents_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                try:
                    file_hash = get_file_hash(file_path)
                    current_files[file_path] = {
                        "hash": file_hash,
                        "name": file,
                        "extension": file_ext,
                        "folder": os.path.relpath(root, documents_folder)
                    }
                except Exception as e:
                    print(f"Error processing file metadata for {file_path}: {e}")
    new_or_changed_files = []
    for file_path, file_info in current_files.items():
        if (file_path not in processed_files_info or 
            processed_files_info[file_path]["hash"] != file_info["hash"]):
            new_or_changed_files.append(file_path)
    new_documents = []
    if new_or_changed_files:
        print(f"Processing {len(new_or_changed_files)} new/changed documents...")
        for file_path in new_or_changed_files:
            try:
                print(f"Processing: {file_path}")
                document_text = extract_text_from_file_path(file_path)
                if document_text.strip():
                    file_info = current_files[file_path]
                    doc = Doc(
                        page_content=document_text,
                        metadata={
                            "source": file_info["name"],
                            "file_path": file_path,
                            "file_type": file_info["extension"],
                            "folder": file_info["folder"]
                        }
                    )
                    new_documents.append(doc)
                    print(f"✅ Successfully processed: {file_info['name']}")
                else:
                    print(f"⚠️ No text extracted from: {file_path}")
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {e}")
                continue
    if new_documents:
        print(f"Adding {len(new_documents)} new documents to vectorstore...")
        # Use optimized chunking strategy for large documents
        print("🔄 Creating optimized chunks...")
        new_doc_chunks = create_optimized_chunks_for_large_docs(new_documents, config_path)
        if existing_vectorstore:
            embedding_model = initialize_embedding_model()
            existing_vectorstore.add_documents(new_doc_chunks)
            final_vectorstore = existing_vectorstore
            print("Added new documents to existing vectorstore")
        else:
            embedding_model = initialize_embedding_model()
            final_vectorstore = FAISS.from_documents(new_doc_chunks, embedding_model)
            print("Created new vectorstore")
        save_vectorstore(final_vectorstore, config_path)
        processed_files_info.update(current_files)
        save_processed_files_info(processed_files_info, config_path)
    else:
        print("No new or changed documents found.")
        final_vectorstore = existing_vectorstore
    total_docs = len(current_files)
    new_docs = len(new_documents)
    processing_info = {
        "total_documents": total_docs,
        "new_documents": new_docs,
        "reused_documents": total_docs - new_docs
    }
    return new_documents, final_vectorstore, processing_info

def clear_vectorstore_and_cache(config_path="config.json"):
    """Clear the vector database and processed files cache to start fresh"""
    vectorstore_path = "vectorstore"
    processed_files_info_path = "processed_files_info.json"
    
    cleared_items = []
    
    # Remove vectorstore files
    if os.path.exists(vectorstore_path):
        try:
            import shutil
            shutil.rmtree(vectorstore_path)
            cleared_items.append("Vector database (FAISS index)")
            print(f"✅ Cleared vector database from {vectorstore_path}")
        except Exception as e:
            print(f"❌ Error clearing vector database: {e}")
    
    # Clear processed files info
    if os.path.exists(processed_files_info_path):
        try:
            # Create empty processed files info
            with open(processed_files_info_path, 'w') as f:
                json.dump({}, f, indent=2)
            cleared_items.append("Processed files cache")
            print(f"✅ Cleared processed files cache")
        except Exception as e:
            print(f"❌ Error clearing processed files info: {e}")
    
    if cleared_items:
        print(f"\n🎯 Successfully cleared: {', '.join(cleared_items)}")
        print("💡 You can now add new documents - they will be processed fresh!")
        return True
    else:
        print("ℹ️ No vector database or cache found to clear")
        return False

def create_optimized_chunks_for_large_docs(documents, config_path="config.json"):
    """
    Improved chunking strategy that works well for documents of all sizes
    Uses intelligent splitting with multiple fallback separators
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Load config
    config = load_config(config_path)
    
    all_chunks = []
    
    for doc in documents:
        doc_text = doc.page_content
        doc_metadata = doc.metadata
        
        # Check document size
        doc_size = len(doc_text)
        print(f"Processing document: {doc_metadata.get('source', 'Unknown')} ({doc_size:,} characters)")
        
        if doc_size > 50000:  # Large document (50k+ chars)
            print("🔍 Large document detected - using multi-tier chunking strategy")
            
            # Multi-tier chunking for large documents
            splitters = [
                # Large chunks for broad context (4000 chars)
                RecursiveCharacterTextSplitter(
                    separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", ". ", ", ", " "],
                    chunk_size=4000,
                    chunk_overlap=400,
                    length_function=len
                ),
                # Medium chunks for detailed content (2000 chars)
                RecursiveCharacterTextSplitter(
                    separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " "],
                    chunk_size=2000,
                    chunk_overlap=300,
                    length_function=len
                ),
                # Small chunks for specific details (1200 chars)
                RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", ", ", " "],
                    chunk_size=1200,
                    chunk_overlap=200,
                    length_function=len
                )
            ]
            
            for i, splitter in enumerate(splitters):
                chunk_type = ["broad", "detailed", "specific"][i]
                chunks = splitter.split_documents([doc])
                
                for j, chunk in enumerate(chunks):
                    enhanced_metadata = {
                        **doc_metadata,
                        "chunk_type": chunk_type,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "doc_size_category": "large",
                        "chunk_size": len(chunk.page_content)
                    }
                    chunk.metadata = enhanced_metadata
                    all_chunks.append(chunk)
                    
        elif doc_size > 5000:  # Medium document (5k-50k chars)
            print("📄 Medium document - using dual-tier chunking")
            
            # Dual-tier chunking for medium documents
            splitters = [
                # Standard chunks (1800 chars)
                RecursiveCharacterTextSplitter(
                    separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " "],
                    chunk_size=1800,
                    chunk_overlap=250,
                    length_function=len
                ),
                # Detailed chunks (1000 chars)
                RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ". ", ", ", " "],
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len
                )
            ]
            
            for i, splitter in enumerate(splitters):
                chunk_type = ["standard", "detailed"][i]
                chunks = splitter.split_documents([doc])
                
                for j, chunk in enumerate(chunks):
                    enhanced_metadata = {
                        **doc_metadata,
                        "chunk_type": chunk_type,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "doc_size_category": "medium",
                        "chunk_size": len(chunk.page_content)
                    }
                    chunk.metadata = enhanced_metadata
                    all_chunks.append(chunk)
                    
        else:  # Small document (under 5k chars)
            print("📋 Small document - using optimized single-tier chunking")
            
            # Smart chunking for small documents
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " "],
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            
            chunks = splitter.split_documents([doc])
            
            # If still only one chunk, try more aggressive splitting
            if len(chunks) == 1 and doc_size > 1000:
                print("🔧 Single chunk detected - applying aggressive splitting")
                aggressive_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n", ". ", "; ", ", ", " ", ""],
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len
                )
                chunks = aggressive_splitter.split_documents([doc])
            
            for j, chunk in enumerate(chunks):
                enhanced_metadata = {
                    **doc_metadata,
                    "chunk_type": "standard",
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "doc_size_category": "small",
                    "chunk_size": len(chunk.page_content)
                }
                chunk.metadata = enhanced_metadata
                all_chunks.append(chunk)
    
    print(f"✅ Total chunks created: {len(all_chunks)}")
    
    # Log chunk distribution for debugging
    chunk_sizes = [len(chunk.page_content) for chunk in all_chunks]
    if chunk_sizes:
        print(f"📊 Chunk size stats - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)//len(chunk_sizes)}")
    
    return all_chunks

def log_retrieved_chunks_for_debugging(query, chunks, enhanced_query=None, log_to_file=True):
    """
    Log retrieved chunks for automation team debugging
    
    Args:
        query (str): Original user query
        chunks (list): List of retrieved chunks (source_documents)
        enhanced_query (str): Enhanced query if used
        log_to_file (bool): Whether to save to debug log file
    """
    import datetime
    
    print("\n" + "="*80)
    print("🔍 CHUNK RETRIEVAL DEBUG INFORMATION")
    print("="*80)
    print(f"⏰ Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"❓ Original Query: {query}")
    if enhanced_query and enhanced_query != query:
        print(f"🔧 Enhanced Query: {enhanced_query}")
    print(f"📊 Total Chunks Retrieved: {len(chunks)}")
    print("-"*80)
    
    chunk_details = []
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata
        content = chunk.page_content
        
        print(f"\n📄 CHUNK {i}:")
        print(f"   📁 Source: {metadata.get('source', 'Unknown')}")
        print(f"   📋 Type: {metadata.get('chunk_type', 'standard')}")
        print(f"   🔢 Index: {metadata.get('chunk_index', 'N/A')}")
        print(f"   📏 Size: {len(content)} characters")
        print(f"   📂 File Path: {metadata.get('file_path', 'N/A')}")
        print(f"   📝 First 100 chars: {content[:100]}...")
        print(f"   📝 Last 100 chars: ...{content[-100:]}")
        
        # Store for file logging
        chunk_info = {
            "chunk_number": i,
            "source": metadata.get('source', 'Unknown'),
            "chunk_type": metadata.get('chunk_type', 'standard'),
            "chunk_index": metadata.get('chunk_index', 'N/A'),
            "size": len(content),
            "file_path": metadata.get('file_path', 'N/A'),
            "content_preview": f"{content[:200]}...{content[-200:]}" if len(content) > 400 else content,
            "full_content": content
        }
        chunk_details.append(chunk_info)
    
    print("="*80)
    
    # Save to debug log file if requested
    if log_to_file:
        try:
            debug_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "original_query": query,
                "enhanced_query": enhanced_query,
                "total_chunks": len(chunks),
                "chunks": chunk_details
            }
            
            import json
            import os
            
            # Create debug logs directory if it doesn't exist
            debug_dir = "debug_logs"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Save with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{debug_dir}/chunk_debug_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(debug_log, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Debug log saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save debug log: {e}")
    
    return chunk_details

def analyze_chunk_coverage(chunks):
    """
    Analyze the coverage and distribution of retrieved chunks for debugging
    """
    print("\n" + "="*60)
    print("📊 CHUNK COVERAGE ANALYSIS")
    print("="*60)
    
    # Group by source document
    sources = {}
    chunk_types = {}
    
    for chunk in chunks:
        source = chunk.metadata.get('source', 'Unknown')
        chunk_type = chunk.metadata.get('chunk_type', 'standard')
        
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
        
        if chunk_type not in chunk_types:
            chunk_types[chunk_type] = 0
        chunk_types[chunk_type] += 1
    
    print("📁 Chunks per Source Document:")
    for source, count in sorted(sources.items()):
        print(f"   • {source}: {count} chunks")
    
    print(f"\n🏷️ Chunks by Type:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"   • {chunk_type}: {count} chunks")
    
    # Calculate coverage statistics
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    
    print(f"\n📏 Size Statistics:")
    print(f"   • Total characters: {total_chars:,}")
    print(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"   • Number of sources: {len(sources)}")
    print(f"   • Number of chunk types: {len(chunk_types)}")
    
    print("="*60)
    
    return {
        "sources": sources,
        "chunk_types": chunk_types,
        "total_chars": total_chars,
        "avg_chunk_size": avg_chunk_size
    }

def test_chunking_logic(config_path="config.json"):
    """
    Test the new chunking logic with existing documents
    """
    print("\n" + "="*60)
    print("🧪 TESTING CHUNKING LOGIC")
    print("="*60)
    
    config = load_config(config_path)
    documents_folder = config.get("documents_folder", "sop_documents")
    
    if not os.path.exists(documents_folder):
        print(f"❌ Documents folder not found: {documents_folder}")
        return
    
    # Get all files in the documents folder
    supported_extensions = ['.pdf', '.docx', '.txt', '.csv', '.xlsx']
    files = []
    for ext in supported_extensions:
        files.extend([f for f in os.listdir(documents_folder) if f.lower().endswith(ext)])
    
    if not files:
        print(f"❌ No supported documents found in {documents_folder}")
        return
    
    print(f"📁 Found {len(files)} document(s) to test:")
    for file in files:
        print(f"   • {file}")
    
    # Process each file and test chunking
    test_documents = []
    for file in files:
        file_path = os.path.join(documents_folder, file)
        print(f"\n🔍 Testing: {file}")
        
        # Extract text
        document_text = extract_text_from_file_path(file_path)
        if document_text.strip():
            doc = Doc(
                page_content=document_text,
                metadata={
                    "source": file,
                    "file_path": file_path,
                    "file_type": os.path.splitext(file)[1].lower(),
                    "folder": ""
                }
            )
            test_documents.append(doc)
            print(f"   ✅ Text extracted: {len(document_text):,} characters")
        else:
            print(f"   ❌ No text extracted")
    
    if test_documents:
        print(f"\n🔄 Testing chunking on {len(test_documents)} document(s)...")
        chunks = create_optimized_chunks_for_large_docs(test_documents, config_path)
        
        print("\n📊 CHUNKING RESULTS:")
        print("-" * 40)
        
        # Group chunks by document
        docs_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in docs_chunks:
                docs_chunks[source] = []
            docs_chunks[source].append(chunk)
        
        for doc_name, doc_chunks in docs_chunks.items():
            print(f"\n📄 {doc_name}:")
            chunk_types = {}
            for chunk in doc_chunks:
                chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                if chunk_type not in chunk_types:
                    chunk_types[chunk_type] = []
                chunk_types[chunk_type].append(len(chunk.page_content))
            
            for chunk_type, sizes in chunk_types.items():
                print(f"   • {chunk_type}: {len(sizes)} chunks, sizes {min(sizes)}-{max(sizes)} chars")
        
        print(f"\n✅ Total chunks: {len(chunks)}")
        return chunks
    else:
        print("❌ No documents processed")
        return []

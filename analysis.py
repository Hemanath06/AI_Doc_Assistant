# analysis.py
import os
import json
from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
# from langchain_community.llms import Grok
from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()

def initialise_llm(config_path="config.json"):
    """
    Initialize LLM with Llama 3 model using config file
    """
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    llm = ChatGroq(
        api_key=config["grok_api_key"],
        model=config["model"]
    )
    return llm


def create_chain(vectorstore):
    llm = initialise_llm()
    
    # OPTIMIZED FOR COMPLETE CONTENT: Get more chunks for full procedures
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance - reduces redundancy
        search_kwargs={
            "k": 30,           # Increased significantly for complete procedures
            "fetch_k": 80,     # More candidates to ensure complete content
            "lambda_mult": 0.6 # More diversity to get different parts of procedures
        }
    )
    print()
    # INTELLIGENT & FLEXIBLE PROMPT: Adapts to any cybersecurity document
    prompt = PromptTemplate(
        template="""You are an experienced cybersecurity analyst. Your job is to find and present ONLY the exact information from the documents that answers the user's question.

Document Context: {context}

Question: {question}

CRITICAL REQUIREMENTS:
1. **ONLY answer based on the provided document context above**
2. **If the specific information is NOT in the documents, you MUST say "No related information is present in the document"**
3. **Do NOT make assumptions or provide general knowledge - stick strictly to what's in the documents**
4. **Match the user's exact question - do not substitute similar terms**
5. **If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'**

INSTRUCTIONS:
1. **Carefully examine the document context** to see if it contains information that directly answers the question
2. **If relevant information exists**: Present it exactly as written in the documents, preserving formatting and structure
3. **If no relevant information exists**: State "No related information is present in the document" 
4. **Do not provide similar but different information** (e.g., don't answer about "password spray" when asked about "pass the ticket")
5. **Maintain document authenticity** - use the exact text, formatting, and structure from the original documents

RESPONSE GUIDELINES:
- Start by checking if the document context actually contains information about what was asked
- If yes: provide the complete relevant information from the documents
- If no: clearly state "No related information is present in the document"
- If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'
- Maintain the original document's numbering, bullet points, and organization
- Keep technical terms, tool names, and specific instructions exactly as written
- Do NOT fill in gaps with general cybersecurity knowledge

Your response:""",
        input_variables=["context", "question"] 
    )

    # KEEP YOUR SIMPLE APPROACH: Same chain type, just enable source docs for debugging
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Changed to True so you can debug what was retrieved
    )

    return chain

def create_chain_with_memory(vectorstore, memory):
    """
    Enhanced conversational chain with smart query processing
    """
    from langchain_classic.chains import ConversationalRetrievalChain
    
    llm = initialise_llm()
    
    # COMPLETE CONTENT RETRIEVAL: Get enough chunks for full procedures
    retriever = vectorstore.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 35,           # Even more chunks for memory-based queries
            "fetch_k": 90,     # More candidates for comprehensive results
            "lambda_mult": 0.6 # More diversity for complete procedures
        }
    )
    
    # Create conversational chain with your proven prompt style
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are an experienced cybersecurity analyst. Your job is to find and present ONLY the exact information from the documents that answers the user's question, considering the conversation history.

Document Context: {context}

Chat History: {chat_history}

Question: {question}

CRITICAL REQUIREMENTS:
1. **ONLY answer based on the provided document context above**
2. **If the specific information is NOT in the documents, you MUST say "No related information is present in the document"**
3. **Do NOT make assumptions or provide general knowledge - stick strictly to what's in the documents**
4. **Match the user's exact question - do not substitute similar terms**
5. **Consider conversation history but still only use document information**
6. **If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'**

DOCUMENT PRIORITIZATION RULES:
- **For vendor/supplier/company questions**: ONLY use information from "Vendor list.pdf" document. Ignore any vendor mentions in alert runbooks.
- **For investigation procedures**: Use information from "Investigation Runbook" or "Alert Runbook" documents.
- **For post-incident steps**: Use information from alert runbooks and investigation procedures.
- **Always prioritize the most relevant document type** for the specific question being asked.

INSTRUCTIONS:
1. **Consider the conversation context** - build upon previous questions and answers when relevant
2. **Carefully examine the document context** to see if it contains information that directly answers the current question
3. **Choose the RIGHT DOCUMENT SOURCE** - if asking about vendors/suppliers/companies, extract ONLY from "Vendor list.pdf", NOT from alert runbooks
4. **If relevant information exists**: Present it exactly as written in the documents, preserving formatting and structure
5. **If no relevant information exists**: State "No related information is present in the document" 
6. **Do not provide similar but different information** (e.g., don't answer about "password spray" when asked about "pass the ticket")
7. **Maintain document authenticity** - use the exact text, formatting, and structure from the original documents

RESPONSE GUIDELINES:
- **VENDOR QUESTIONS**: Extract information EXCLUSIVELY from "Vendor list.pdf" - ignore any vendor mentions in other documents
- Start by checking if the document context actually contains information about what was asked
- If yes: provide the complete relevant information from the documents with proper formatting
- If no: clearly state "No related information is present in the document"
- If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'
- Maintain the original document's numbering, bullet points, and organization
- Keep technical terms, tool names, and specific instructions exactly as written
- Do NOT fill in gaps with general cybersecurity knowledge
- Reference previous conversation when it adds context to the current answer

Your response:""",
                input_variables=["context", "chat_history", "question"]
            )
        }
    )
    return chain

def enhance_query_for_better_retrieval(question):
    """
    More conservative query enhancement that preserves exact user intent
    """
    question_lower = question.lower()
    
    # Only enhance if we're confident it won't change the meaning
    enhanced_terms = []
    
    # Very minimal enhancement - only add terms that clarify context without changing meaning
    if any(word in question_lower for word in ["after", "post", "following"]) and "incident" not in question_lower:
        enhanced_terms.append("post-incident")
    
    # Only add "investigation" if they're clearly asking about investigating something
    if any(word in question_lower for word in ["investigate", "analysis", "review"]) and "investigation" not in question_lower:
        enhanced_terms.append("investigation") 
    
    # Be very conservative - only add one enhancement term max to avoid confusion
    if enhanced_terms:
        enhanced_query = f"{question} {enhanced_terms[0]}"  # Only use first term
        print(f"🔍 Conservatively enhanced query: {enhanced_query}")
        return enhanced_query
    
    print(f"🔍 Using original query (no enhancement needed): {question}")
    return question

def validate_chunk_relevance(query, chunks, threshold=0.3):
    """
    Validate if retrieved chunks are actually relevant to the user's query
    Returns True if chunks seem relevant, False otherwise
    """
    if not chunks:
        return False
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove common words that don't indicate relevance
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'how', 'what', 'when', 'where', 'why', 'do', 'does', 'did', 'can', 'will', 'would', 'should'}
    meaningful_query_words = query_words - stop_words
    
    if not meaningful_query_words:
        return True  # If no meaningful words, proceed with retrieval
    
    # Check if any chunk contains meaningful overlap with the query
    for chunk in chunks:
        chunk_text = chunk.page_content.lower()
        chunk_words = set(chunk_text.split())
        
        # Calculate overlap ratio
        overlap = meaningful_query_words.intersection(chunk_words)
        overlap_ratio = len(overlap) / len(meaningful_query_words)
        
        if overlap_ratio >= threshold:
            print(f"✅ Found relevant chunk with {overlap_ratio:.2f} relevance score")
            return True
    
    print(f"⚠️ No chunks seem relevant to query: '{query}'")
    print(f"   Query keywords: {meaningful_query_words}")
    return False

def create_enhanced_chain_with_validation(vectorstore, memory):
    """
    Enhanced chain that validates chunk relevance before generating response
    """
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_core.runnables import RunnableLambda
    
    llm = initialise_llm()
    
    # COMPLETE CONTENT RETRIEVAL: Get enough chunks for full procedures
    retriever = vectorstore.as_retriever(
        search_type="mmr",  
        search_kwargs={
            "k": 35,           # Even more chunks for memory-based queries
            "fetch_k": 90,     # More candidates for comprehensive results
            "lambda_mult": 0.6 # More diversity for complete procedures
        }
    )
    
    # Custom validation wrapper for the retriever
    def validated_retrieval(query):
        chunks = retriever.get_relevant_documents(query)
        
        # Validate if chunks are actually relevant
        if not validate_chunk_relevance(query, chunks):
            print("🚫 Chunks don't seem relevant - returning empty context")
            return []  # Return empty to trigger "no information" response
        
        return chunks
    
    # Create conversational chain with validation
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,  # We'll handle validation in main.py
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are an experienced cybersecurity analyst. Your job is to find and present ONLY the exact information from the documents that answers the user's question, considering the conversation history.

Document Context: {context}

Chat History: {chat_history}

Question: {question}

CRITICAL REQUIREMENTS:
1. **ONLY answer based on the provided document context above**
2. **If the specific information is NOT in the documents, you MUST say "No related information is present in the document"**
3. **Do NOT make assumptions or provide general knowledge - stick strictly to what's in the documents**
4. **Match the user's exact question - do not substitute similar terms**
5. **Consider conversation history but still only use document information**
6. **If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'**

DOCUMENT PRIORITIZATION RULES:
- **For vendor/supplier/company questions**: ONLY use information from "Vendor list.pdf" document. Ignore any vendor mentions in alert runbooks.
- **For investigation procedures**: Use information from "Investigation Runbook" or "Alert Runbook" documents.
- **For post-incident steps**: Use information from alert runbooks and investigation procedures.
- **Always prioritize the most relevant document type** for the specific question being asked.

INSTRUCTIONS:
1. **Consider the conversation context** - build upon previous questions and answers when relevant
2. **Carefully examine the document context** to see if it contains information that directly answers the current question
3. **Choose the RIGHT DOCUMENT SOURCE** - if asking about vendors/suppliers/companies, extract ONLY from "Vendor list.pdf", NOT from alert runbooks
4. **If relevant information exists**: Present it exactly as written in the documents, preserving formatting and structure
5. **If no relevant information exists**: State "No related information is present in the document" 
6. **Do not provide similar but different information** (e.g., don't answer about "password spray" when asked about "pass the ticket")
7. **Maintain document authenticity** - use the exact text, formatting, and structure from the original documents

RESPONSE GUIDELINES:
- **VENDOR QUESTIONS**: Extract information EXCLUSIVELY from "Vendor list.pdf" - ignore any vendor mentions in other documents
- Start by checking if the document context actually contains information about what was asked
- If yes: provide the complete relevant information from the documents with proper formatting
- If no: clearly state "No related information is present in the document"
- If the question is like 'Is X a vendor?' and X is not present in the vendor list, display 'No, X is not a vendor.'
- Maintain the original document's numbering, bullet points, and organization
- Keep technical terms, tool names, and specific instructions exactly as written
- Do NOT fill in gaps with general cybersecurity knowledge
- Reference previous conversation when it adds context to the current answer

Your response:""",
                input_variables=["context", "chat_history", "question"]
            )
        }
    )
    return chain

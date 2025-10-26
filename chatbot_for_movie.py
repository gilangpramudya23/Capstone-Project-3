import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import ToolMessage

# CONFIGURATION

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

collection_name = "product_documents"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# TOOLS

@tool
def search_movies_tool(question):
    """Use this tool to find the most relevant movies based on a user's query.
    You can search by Movie title, Actor, and Director.
    Return movies that best match the user's intent, even if the query is phrased naturally."""
    results = qdrant.similarity_search(question, k=10)
    
    # Format the results into readable text
    formatted = []
    for i, doc in enumerate(results, 1):
        metadata = doc.metadata
        formatted.append(
            f"{i}. **{metadata.get('title', 'N/A')}** ({metadata.get('released_year', 'N/A')})\n"
            f"   - Rating: {metadata.get('rating', 'N/A')}/10\n"
            f"   - Genre: {metadata.get('genre', 'N/A')}\n"
            f"   - Director: {metadata.get('director', 'N/A')}\n"
            f"   - Stars: {metadata.get('star1', 'N/A')}, {metadata.get('star2', 'N/A')}\n"
            f"   - Overview: {metadata.get('overview', 'N/A')[:200]}...\n"
        )
    return "\n".join(formatted) if formatted else "No movies found."

@tool
def recommend_movies_tool(question):
    """Use this tool to get movie recommendations based on a movie the user likes.
    The input can be a movie title or a short description of the type of movie they liked."""
    results = qdrant.similarity_search(question, k=10)
    
    # Format recommendations
    recommendations = []
    for i, doc in enumerate(results[1:], 1):
        metadata = doc.metadata
        recommendations.append(
            f"{i}. **{metadata.get('title', 'N/A')}** ({metadata.get('released_year', 'N/A')})\n"
            f"   - Rating: {metadata.get('rating', 'N/A')}/10\n"
            f"   - Genre: {metadata.get('genre', 'N/A')}\n"
            f"   - Why: Similar themes, style, and tone\n"
        )
    return "\n".join(recommendations) if recommendations else "No recommendations found."

@tool
def statistics_tool(question):
    """Use this tool to get top rated movies, best movies by genre, or year analysis.
    Return a ranked list or a brief analysis summary based on the query."""
    results = qdrant.similarity_search(question, k=50)

    # Fallback: if no results, get a broader set
    if not results:
        results = qdrant.similarity_search("movie", k=100)

    # Sort by rating
    sorted_results = sorted(
        results,
        key=lambda x: float(x.metadata.get('rating', 0) or 0),
        reverse=True
    )

    stats = []
    for i, doc in enumerate(sorted_results[:10], 1):
        metadata = doc.metadata
        stats.append(
            f"{i}. **{metadata.get('title', 'N/A')}** ({metadata.get('released_year', 'N/A')})\n"
            f"   - Rating: {metadata.get('rating', 'N/A')}/10\n"
            f"   - Genre: {metadata.get('genre', 'N/A')}\n"
            f"   - Metascore: {metadata.get('meta_score', 'N/A')}\n"
        )
    return "\n".join(stats) if stats else "No statistics available."

@tool
def compare_movies_tool(question):
    """Use this tool to compare multiple movies side by side.
    Focus on similarities and differences.
    Return a concise comparison summary or table-style result if needed."""
    results = qdrant.similarity_search(question, k=10)
    
    # Format comparisons
    comparisons = []
    for doc in results[:5]:  # Limit to 5 movies for comparison
        metadata = doc.metadata
        comparisons.append(
            f"**{metadata.get('title', 'N/A')}**\n"
            f"  - Year: {metadata.get('released_year', 'N/A')}\n"
            f"  - Rating: {metadata.get('rating', 'N/A')}/10\n"
            f"  - Genre: {metadata.get('genre', 'N/A')}\n"
            f"  - Director: {metadata.get('director', 'N/A')}\n"
            f"  - Runtime: {metadata.get('duration', 'N/A')}\n"
            f"  - Gross: ${metadata.get('gross', 'N/A')}\n"
        )
    return "\n---\n".join(comparisons) if comparisons else "No movies found to compare."

# MAIN FUNCTION

def chat_movie(question, history):
    # Build messages history
    messages = []
    for msg in history:
        if msg["role"] == "Human":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "AI":
            messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    
    # Process with supervisor
    result_messages = []
    agents_used = []
    
    for chunk in supervisor.stream({"messages": messages}, stream_mode="values"):
        if "messages" in chunk:
            result_messages = chunk["messages"]
            last_message = chunk["messages"][-1]
            
            if hasattr(last_message, 'name') and last_message.name:
                if last_message.name not in agents_used:
                    agents_used.append(last_message.name)
    
    answer = result_messages[-1].content
    result = {"messages": result_messages}
    
    # Calculate tokens and price
    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    # Extract tool messages
    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(str(message.content))

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages,
        "agents_used": agents_used
    }
    return response

# SPECIALIST AGENTS

# Search Agent
search_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_movies_tool],
    prompt="You are a movie search specialist." \
    "Your goal is to help users find the most relevant movies based on their questions." \
    "Use the 'search_movies_tool' to search by title, actor, director, or descriptive keywords." \
    "Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="search_agent"
)

# Recommendation Agent
recommendation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[recommend_movies_tool],
    prompt="You are a movie recommendation specialist." \
    "Suggest movies similar to the one the user liked by analyzing genre, tone, themes, director style, or audience appeal. " \
    "Use the 'recommend_movies_tool' to find up to 10 relevant movies and present them in a clear, ranked list with brief descriptions." \
    "Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="recommendation_agent"
)

# Statistics Agent
statistics_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[statistics_tool],
    prompt="You are a movie statistics specialist. Provide top-rated movies, best by genre, or year analysis." \
    "Provide results as a ranked list or short summary, including relevant metrics like ratings or year." \
    "Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="statistics_agent"
)

# Comparison Agent
comparison_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[compare_movies_tool],
    prompt="You are a movie comparison specialist. Compare multiple movies and highlight similarities and differences." \
    "Use the 'compare_movies_tool' to analyze and compare two or more movies." \
    "Present your findings in a structured summary or bullet list for easy understanding." \
    "Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="comparison_agent"
)

# SUPERVISOR AGENT

supervisor = create_supervisor(
    agents=[search_agent, recommendation_agent, statistics_agent, comparison_agent],
    model=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    prompt="""You are the Supervisor Agent that manages four movie specialists.
    Your goal is to correctly identify the user’s intent and route the query to the most suitable specialist agent.
    Available agents:
    - search_agent → Finds movies by title, actor, director, or keywords.
    - recommendation_agent → Suggests similar movies based on user preferences or liked titles.
    - statistics_agent → Provides top-rated movies, best-by-genre lists, or year-based analysis.
    - comparison_agent → Compares two or more movies side by side.
    
    Route each question to the best specialist.
    Your audiences are movie enthusiasts seeking accurate and relevant information."""
).compile()

# STREAMLIT APP

st.title("🎬 Looking for Something to Watch?")
st.caption("Grab your snacks! I’ll help you find the perfect pick.")

import datetime

hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif hour < 18:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

st.subheader(f"{greeting}! Got a movie in mind or need some inspiration?")

# Display header image if exists
try:
    st.image("./Recipe Master Agent/header_img.png")
except:
    pass

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_info" in message and message["agent_info"]:
            st.caption(message["agent_info"])

# Chat input
if prompt := st.chat_input("Ask me about movies!"):
    messages_history = st.session_state.get("messages", [])[-20:]
    
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    with st.chat_message("AI"):
        with st.spinner("Processing with specialist agents..."):
            response = chat_movie(prompt, messages_history)
            answer = response["answer"]
            agents_used = response.get('agents_used', [])
            
            st.markdown(answer)
            
            # Show which agents handled the query
            agent_info = f"🤖 Handled by: **{', '.join(agents_used)}**"
            st.caption(agent_info)
            
            st.session_state.messages.append({
                "role": "AI",
                "content": answer,
                "agent_info": agent_info
            })
    
    # Expandable details
    with st.expander("**Tool Calls:**"):
        st.code(response["tool_messages"])
    
    with st.expander("**History Chat:**"):
        history_display = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
        st.code(history_display if history_display else "No history")
    
    with st.expander("**Usage Details:**"):
        st.code(
            f'Input tokens: {response["total_input_tokens"]}\n'
            f'Output tokens: {response["total_output_tokens"]}\n'
            f'Price: Rp {response["price"]:.4f}'
        )

# Sidebar examples
with st.sidebar:
    st.header("💡 Example Queries")

    def send_query_to_chat(query_text):
        messages_history = st.session_state.get("messages", [])[-20:]
        st.session_state.messages.append({"role": "Human", "content": query_text})

        response = chat_movie(query_text, messages_history)
        agents_used = response.get('agents_used', [])
        agent_info = f"🤖 Handled by: **{', '.join(agents_used)}**"

        st.session_state.messages.append({
            "role": "AI",
            "content": response["answer"],
            "agent_info": agent_info
        })

        st.rerun()
    
    st.subheader("🔍 Search")
    if st.button("Find Nolan films", use_container_width=True):
        st.session_state.next_query = "Find movies directed by Christopher Nolan"
        st.rerun()
    
    st.subheader("🎯 Recommendations")
    if st.button("Movies like Inception", use_container_width=True):
        st.session_state.next_query = "Recommend movies like Inception"
        st.rerun()
    
    st.subheader("📊 Statistics")
    if st.button("Top 10 rated", use_container_width=True):
        st.session_state.next_query = "What are the top 10 highest rated movies?"
        st.rerun()
    
    st.subheader("⚖️ Compare")
    if st.button("Compare movie 1 & 2", use_container_width=True):
        st.session_state.next_query = "Compare Movie 1 and Movie 2"
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("**🤖 Specialist Agents:**")
    st.caption("• Search Agent")
    st.caption("• Recommendation Agent")
    st.caption("• Statistics Agent")
    st.caption("• Comparison Agent")

# Handle example queries
if "next_query" in st.session_state:
    query = st.session_state.next_query
    del st.session_state.next_query
    
    messages_history = st.session_state.get("messages", [])[-20:]
    st.session_state.messages.append({"role": "Human", "content": query})
    
    response = chat_movie(query, messages_history)
    agents_used = response.get('agents_used', [])
    agent_info = f"🤖 Handled by: **{', '.join(agents_used)}**"
    
    st.session_state.messages.append({
        "role": "AI",
        "content": response["answer"],
        "agent_info": agent_info
    })

    st.rerun()



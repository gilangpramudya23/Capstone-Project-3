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
def get_relevant_docs(question):
    """Use this tool to get relevant documents about movies."""
    results = qdrant.similarity_search(question, k=10)
    return results

@tool
def search_movies(query: str) -> str:
    """Search for movies by title, actor, director, or keywords."""
    results = qdrant.similarity_search(query, k=5)
    formatted = []
    for doc in results:
        metadata = doc.metadata
        formatted.append(
            f"**{metadata.get('Series_Title', 'N/A')}** ({metadata.get('Released_Year', 'N/A')})\n"
            f"Rating: {metadata.get('IMDB_Rating', 'N/A')}/10 | Genre: {metadata.get('Genre', 'N/A')}\n"
            f"Director: {metadata.get('Director', 'N/A')}\n"
            f"Stars: {metadata.get('Star1', 'N/A')}, {metadata.get('Star2', 'N/A')}\n"
        )
    return "\n---\n".join(formatted)

@tool
def get_recommendations(movie_title: str) -> str:
    """Get movie recommendations similar to the given movie title."""
    results = qdrant.similarity_search(movie_title, k=6)
    recommendations = []
    for i, doc in enumerate(results[1:], 1):
        metadata = doc.metadata
        recommendations.append(
            f"{i}. {metadata.get('Series_Title', 'N/A')} ({metadata.get('Released_Year', 'N/A')}) - "
            f"Rating: {metadata.get('IMDB_Rating', 'N/A')}/10"
        )
    return "\n".join(recommendations)

@tool
def statistics_tool(IMDB_Rating: str) -> str:
    """Use this tool to get top-rated movies, best movies by genre, or year analysis.
    It detects genre and rating conditions automatically and returns ranked results."""
    
    question_lower = question.lower()
    genre_detected = None
    genres = [
        "action", "drama", "comedy", "thriller", "romance",
        "horror", "sci-fi", "adventure", "animation", "crime"
    ]
    for g in genres:
        if g in question_lower:
            genre_detected = g.title()
            break

    # Run the base similarity search with a wider range
    results = qdrant.similarity_search(question, k=10)
    if not results:
        return "No movies found for your query."

    # Filter by detected genre
    if genre_detected:
        results = [r for r in results if genre_detected in str(r.metadata.get("genre", ""))]

    # Sort results by rating
    sorted_results = sorted(
        results,
        key=lambda x: float(x.metadata.get("rating", 0) or 0),
        reverse=True
    )

    # Take the top 10
    top_movies = sorted_results[:10]

    if not top_movies:
        return f"No {genre_detected or ''} movies found with sufficient rating."

    # Format the output neatly
    formatted = []
    for i, doc in enumerate(top_movies, 1):
        m = doc.metadata
        formatted.append(
            f"{i}. **{m.get('title', 'N/A')}** ({m.get('released_year', 'N/A')})\n"
            f"   - Rating: {m.get('rating', 'N/A')}/10\n"
            f"   - Genre: {m.get('genre', 'N/A')}\n"
            f"   - Director: {m.get('director', 'N/A')}\n"
            f"   - Stars: {m.get('star1', 'N/A')}, {m.get('star2', 'N/A')}\n"
        )

    return (
        f"🎬 **Top 10 Rated {genre_detected or 'Movies'}:**\n\n"
        + "\n".join(formatted)
    )

@tool
def compare_movies(movie_titles: str) -> str:
    """Compare multiple movies. Input should be comma-separated titles."""
    titles = [t.strip() for t in movie_titles.split(',')]
    comparisons = []
    for title in titles:
        results = qdrant.similarity_search(title, k=1)
        if results:
            doc = results[0]
            metadata = doc.metadata
            comparisons.append(
                f"{metadata.get('Series_Title', 'N/A')}: "
                f"Rating {metadata.get('IMDB_Rating', 'N/A')}, "
                f"Year {metadata.get('Released_Year', 'N/A')}, "
                f"Genre {metadata.get('Genre', 'N/A')}"
            )
    return "\n".join(comparisons)

# chat_movie FUNCTION

def chat_movie(question, history, tools_list, agent_prompt):
    agent = create_react_agent(
        model=llm,
        tools=tools_list,
        prompt=agent_prompt
    )

    messages = []
    # Add conversation history
    for msg in history:
        if msg["role"] == "Human":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "AI":
            messages.append({"role": "assistant", "content": msg["content"]})
    
    messages.append({"role": "user", "content": question})

    result = agent.invoke(
        {"messages": messages}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

# SPECIALIST AGENTS (From chat_movie function!)

# Search Agent
search_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[search_movies, get_relevant_docs],
    prompt="You are a movie search specialist. Find movies by title, actor, director, or keywords. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="search_agent"
)

# Recommendation Agent
recommendation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_recommendations],
    prompt="You are a movie recommendation specialist. Suggest similar movies and explain why. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="recommendation_agent"
)

# Statistics Agent
statistics_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[statistics_tool],
    prompt="You are a movie statistics specialist. Provide top-rated movies and analyze trends.",
    name="statistics_agent"
)

# Comparison Agent - uses your chat_chef function
comparison_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[compare_movies],
    prompt="You are a movie comparison specialist. Compare movies side-by-side. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="comparison_agent"
)

# SUPERVISOR

# Initialize supervisor
USE_SUPERVISOR = st.sidebar.checkbox("🤖 Enable Supervisor Mode", value=True)

if USE_SUPERVISOR:
    supervisor = create_supervisor(
        agents=[search_agent, recommendation_agent, statistics_agent, comparison_agent],
        model=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
        prompt="""You manage four movie specialists:
        - search_agent: Finds movies by title/actor/director
        - recommendation_agent: Recommends similar movies
        - comparison_agent: Compares multiple movies
        
        Route queries to the most appropriate specialist."""
    ).compile()

# WRAPPER FUNCTION

def process_question(question, history):
    
    if USE_SUPERVISOR:
        # Use supervisor mode
        messages = []
        for msg in history:
            role = "user" if msg["role"] == "Human" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        
        full_response = ""
        agents_used = []
        total_input_tokens = 0
        total_output_tokens = 0
        tool_messages = []
        
        for chunk in supervisor.stream({"messages": messages}, stream_mode="values"):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                
                if hasattr(last_message, 'name') and last_message.name:
                    if last_message.name not in agents_used:
                        agents_used.append(last_message.name)
                
                if hasattr(last_message, 'content'):
                    full_response = last_message.content
                
                if hasattr(last_message, 'response_metadata'):
                    metadata = last_message.response_metadata
                    if "usage_metadata" in metadata:
                        total_input_tokens += metadata["usage_metadata"].get("input_tokens", 0)
                        total_output_tokens += metadata["usage_metadata"].get("output_tokens", 0)
                
                if isinstance(last_message, ToolMessage):
                    tool_messages.append(last_message.content)
        
        price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
        
        return {
            "answer": full_response,
            "price": price,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_messages": tool_messages,
            "agents_used": agents_used
        }
    
    else:
        tools = [get_relevant_docs]
        prompt = "You are a movie domain expert with deep knowledge of films, directors, actors, genres, and industry trends. Respond to questions related to movies. "
        
        result = chat_movie(question, history, tools, prompt)
        result["agents_used"] = ["single_agent"]
        return result

# STREAMLIT APP

st.title("🎬 Looking for Something to Watch?")

# Mode indicator
if USE_SUPERVISOR:
    st.caption("🤖 Running in **Supervisor Mode** with 4 specialist agents")
else:
    st.caption("Grab your snacks, I’ll help you find the perfect pick!")

# Display header image if exists
try:
    st.image("./Recipe Master Agent/header_img.png")
except:
    pass

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_info" in message and message["agent_info"]:
            st.caption(message["agent_info"])

# Accept user input
if prompt := st.chat_input("Ask me movies question"):
    messages_history = st.session_state.get("messages", [])[-20:]
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response
    with st.chat_message("AI"):
        with st.spinner("Processing..."):
            response = process_question(prompt, messages_history)
            answer = response["answer"]
            
            st.markdown(answer)
            
            # Show agent info
            if USE_SUPERVISOR and "agents_used" in response:
                agent_info = f"🤖 Handled by: **{', '.join(response['agents_used'])}**"
                st.caption(agent_info)
                st.session_state.messages.append({
                    "role": "AI",
                    "content": answer,
                    "agent_info": agent_info
                })
            else:
                st.session_state.messages.append({"role": "AI", "content": answer})
    
    # Show details in expanders
    with st.expander("**Tool Calls:**"):
        if response["tool_messages"]:
            for i, msg in enumerate(response["tool_messages"], 1):
                # Format the tool message nicely
                if isinstance(msg, str):
                    st.code(msg, language="python")
                else:
                    st.code(str(msg), language="python")
        else:
            st.info("No tools were called")
    
    with st.expander("**History Chat:**"):
        history_display = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
        st.code(history_display if history_display else "No history")
    
    with st.expander("**Usage Details:**"):
        st.code(
            f'Input tokens: {response["total_input_tokens"]}\n'
            f'Output tokens: {response["total_output_tokens"]}\n'
            f'Price: Rp {response["price"]:.4f}'
        )

# Sidebar
with st.sidebar:
    st.header("💡 Example Queries")
    
    st.subheader("🔍 Search")
    if st.button("Find Nolan films"):
        st.session_state.next_query = "Find movies directed by Christopher Nolan"
        st.rerun()
    
    st.subheader("🎯 Recommendations")
    if st.button("Movies like Inception"):
        st.session_state.next_query = "Recommend movies like Inception"
        st.rerun()
    
    st.subheader("📊 Rated Movie")
    if st.button("Top 10 rated action movies"):
        st.session_state.next_query = "Top 10 rated action movies"
        st.rerun()
    
    st.subheader("⚖️ Compare")
    if st.button("Compare Godfather 1 & 2"):
        st.session_state.next_query = "Compare The Godfather and The Godfather Part II"
        st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Handle example queries
if "next_query" in st.session_state:
    query = st.session_state.next_query
    del st.session_state.next_query
    
    messages_history = st.session_state.get("messages", [])[-20:]
    st.session_state.messages.append({"role": "Human", "content": query})
    
    response = process_question(query, messages_history)
    agent_info = ""
    if USE_SUPERVISOR and "agents_used" in response:
        agent_info = f"🤖 Handled by: **{', '.join(response['agents_used'])}**"
    
    st.session_state.messages.append({
        "role": "AI",
        "content": response["answer"],
        "agent_info": agent_info
    })
    st.rerun()

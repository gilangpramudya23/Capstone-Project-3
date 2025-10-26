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
            f"**{metadata.get('title', 'N/A')}** ({metadata.get('released_year', 'N/A')})\n"
            f"Rating: {metadata.get('rating', 'N/A')}/10 | Genre: {metadata.get('genre', 'N/A')}\n"
            f"Director: {metadata.get('director', 'N/A')}\n"
            f"Stars: {metadata.get('star1', 'N/A')}, {metadata.get('star2', 'N/A')}\n"
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
            f"{i}. {metadata.get('title', 'N/A')} ({metadata.get('released_year', 'N/A')}) - "
            f"Rating: {metadata.get('rating', 'N/A')}/10"
        )
    return "\n".join(recommendations)

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
                f"{metadata.get('title', 'N/A')}: "
                f"Genre {metadata.get('genre', 'N/A')}"
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

# SPECIALIST AGENTS
tools = [search_movies, get_relevant_docs, get_recommendations, compare_movies]

# Search Agent
search_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt="You are a movie search specialist. Find movies by title, actor, director, or keywords. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="search_agent"
)

# Recommendation Agent
recommendation_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt="You are a movie recommendation specialist. Suggest similar movies and explain why. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="recommendation_agent"
)

# Comparison Agent - uses your chat_chef function
comparison_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt="You are a movie comparison specialist. Compare movies side-by-side. Be informative, friendly, and insightful. Avoid opinions or speculation.",
    name="comparison_agent"
)

# SUPERVISOR

# Initialize supervisor
USE_SUPERVISOR = st.sidebar.checkbox("🤖 Enable Supervisor Mode", value=True)
if USE_SUPERVISOR:
    supervisor = create_supervisor(
        agents=[search_agent, recommendation_agent, comparison_agent],
        model=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
        prompt="""You manage four movie specialists:
        - search_agent: Finds movies by title/actor/director
        - recommendation_agent: Recommends similar movies
        - comparison_agent: Compares multiple movies
        
        Route queries to the most appropriate specialist.
        Show the answer result to user."""
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
    st.caption("Grab your snacks, 4 Agents help you find the perfect pick!")
else:
    st.caption("Grab your snacks, 4 Agents help you find the perfect pick!")

# Display header image if exists
try:
    st.image("./Recipe Master Agent/header_img.png")
except:
    pass

import datetime

hour = datetime.datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif hour < 18:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

st.subheader(f"{greeting}! How can I help you find your next great movie?")
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
if prompt := st.chat_input("Ask me anything about films search, recommend, compare, or anything!"):
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
    if st.button("Find movies directed by David Fincher"):
        st.session_state.next_query = "Find movies directed by David Fincher"
        st.rerun()

    if st.button("Find three movies released between 2010 and 2012"):
        st.session_state.next_query = "Find three movies released between 2010 and 2012"
        st.rerun()

    if st.button("Find three movies with ratings between 8 and 10."):
        st.session_state.next_query = "Find three movies with ratings between 8 and 10."
        st.rerun()
    
    st.subheader("🎯 Recommendations")
    if st.button("Recommend movies like Inception"):
        st.session_state.next_query = "Recommend movies like Inception"
        st.rerun()

    if st.button("Top 5 action movies"):
        st.session_state.next_query = "Top 5 action movies"
        st.rerun()

    if st.button("Top 5 highest-rated action movies"):
        st.session_state.next_query = "Top 5 highest-rated action movies"
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










#!/usr/bin/env python3
"""
Streamlit Frontend for MCP Coordinator
Interactive web interface for querying the database system
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import time
import json

# Optional plotly imports for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Load environment variables
load_dotenv()

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our coordinator
try:
    from coordinator.mcp_coordinator import MCPCoordinator
    COORDINATOR_AVAILABLE = True
except ImportError as e:
    COORDINATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title = "MCP Database Coordinator",
    page_icon = "🤖",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .coordinator-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .retrieval-message {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .report-message {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .error-message {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .system-status {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html = True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = None
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'initialization_attempted' not in st.session_state:
        st.session_state.initialization_attempted = False

def setup_coordinator():
    """Initialize the MCP Coordinator"""
    if not COORDINATOR_AVAILABLE:
        st.error(f"Cannot import coordinator: {IMPORT_ERROR}")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment variables")
        st.info("Please set your OpenAI API key in the .env file")
        return None
    
    try:
        coordinator = MCPCoordinator(
            openai_api_key = api_key,
            db_path = "output/data/demo.db",
            faiss_index_path = "output/data/demo.index",
            model = "gpt-4o-mini"
        )
        st.session_state.api_key_valid = True
        return coordinator
    except Exception as e:
        st.error(f"Failed to initialize coordinator: {str(e)}")
        return None

def display_message(message, message_type = "user"):
    """Display a chat message with appropriate styling"""
    css_class = f"{message_type}-message"
    
    if message_type == "user":
        title = "You"
    elif message_type == "coordinator":
        title = "Coordinator"
    elif message_type == "retrieval":
        title = "Retrieval Agent"
    elif message_type == "report":
        title = "Report Agent"
    elif message_type == "error":
        title = "Error"
    else:
        title = "System"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{title}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html = True)

def parse_report_data(content):
    """Parse report content to extract JSON data"""
    if "WEB_REPORT_JSON::" in content:
        json_part = content.split("WEB_REPORT_JSON::", 1)[1]
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            return None
    return None

def display_report_dashboard(report_data):
    """Display a comprehensive report dashboard"""
    if "error" in report_data:
        st.error(f"{report_data['error']}")
        return
    
    # Report header
    st.markdown(f"## {report_data['title']}")
    st.markdown(f"*Generated on: {report_data['generated_on']}*")
    
    if report_data['filters_applied']:
        st.info(f"**Filters Applied:** {report_data['filters_applied']}")
    
    # Key metrics
    stats = report_data['summary_stats']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("Total Cost", f"${stats['total_cost']:,.2f}")
    with col3:
        st.metric("Average Cost", f"${stats['avg_cost']:.2f}")
    with col4:
        st.metric("Approval Rate", f"{stats['approval_rate']:.1f}%")
    
    # Top analysis charts
    st.markdown("---")
    st.markdown("### Top Analysis")
    
    top_analysis = report_data.get('top_analysis', {})
    
    # Create columns for charts
    num_charts = len(top_analysis)
    if num_charts > 0:
        if num_charts == 1:
            cols = [st.container()]
        elif num_charts == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(min(3, num_charts))
        
        chart_idx = 0
        for category, data in top_analysis.items():
            if data and chart_idx < len(cols):
                with cols[chart_idx]:
                    # Create chart data
                    df_chart = pd.DataFrame({
                        'Item': list(data.keys()),
                        'Count': list(data.values())
                    })
                    
                    st.markdown(f"**Top {category.replace('_', ' ').title()}**")
                    
                    if PLOTLY_AVAILABLE:
                        # Create interactive bar chart
                        fig = px.bar(
                            df_chart, 
                            x = 'Count', 
                            y = 'Item',
                            orientation = 'h',
                            title = f"Top {category.replace('_', ' ').title()}",
                            color = 'Count',
                            color_continuous_scale = 'Blues'
                        )
                        fig.update_layout(height = 300, showlegend = False)
                        st.plotly_chart(fig, use_container_width = True)
                    else:
                        # Fallback to Streamlit bar chart
                        st.bar_chart(df_chart.set_index('Item'))
                chart_idx += 1
    
    # Group analysis (if available)
    if report_data.get('group_analysis'):
        st.markdown("---")
        st.markdown(f"### Analysis by {report_data['group_analysis']['group_by'].title()}")
        
        group_data = report_data['group_analysis']['data']
        df_group = pd.DataFrame(group_data)
        
        # Display as table
        st.dataframe(df_group, use_container_width = True)
        
        # Create visualization
        if len(df_group) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Return Count by Group**")
                if PLOTLY_AVAILABLE:
                    # Interactive chart
                    fig1 = px.bar(
                        df_group, 
                        x = 'group', 
                        y = 'return_count',
                        title = "Return Count by Group",
                        color = 'return_count',
                        color_continuous_scale = 'Reds'
                    )
                    fig1.update_layout(height = 400)
                    st.plotly_chart(fig1, use_container_width = True)
                else:
                    # Fallback to Streamlit chart
                    st.bar_chart(df_group.set_index('group')['return_count'])
            
            with col2:
                # Approval rate chart (if available)
                if 'approval_rate' in df_group.columns:
                    st.markdown("**Approval Rate by Group (%)**")
                    if PLOTLY_AVAILABLE:
                        # Interactive chart
                        fig2 = px.bar(
                            df_group, 
                            x = 'group', 
                            y = 'approval_rate',
                            title = "Approval Rate by Group (%)",
                            color = 'approval_rate',
                            color_continuous_scale = 'Greens'
                        )
                        fig2.update_layout(height = 400)
                        st.plotly_chart(fig2, use_container_width = True)
                    else:
                        # Fallback to Streamlit chart
                        st.bar_chart(df_group.set_index('group')['approval_rate'])
    
    # Raw data table
    st.markdown("---")
    st.markdown(f"### Raw Data ({report_data['total_raw_records']} total records)")
    
    if report_data['raw_data']:
        df_raw = pd.DataFrame(report_data['raw_data'])
        
        # Show data with pagination
        if len(df_raw) > 0:
            st.dataframe(df_raw, use_container_width = True, height = 400)
            
            if report_data['total_raw_records'] > len(df_raw):
                st.info(f"Showing first {len(df_raw)} of {report_data['total_raw_records']} records")
        else:
            st.info("No data available to display")
    else:
        st.info("No raw data available")

def process_coordinator_result(result):
    """Process and display coordinator results"""
    if not result["success"]:
        display_message(f"Error: {result.get('error', 'Unknown error')}", "error")
        return
    
    # Display agent summary
    agents_used = result.get('agents_used', [])
    total_messages = result.get('total_messages', 0)
    final_status = result.get('final_status', 'unknown')
    
    st.markdown(f"""
    <div class="system-status">
        <strong>📋 Processing Summary:</strong><br>
        • Status: {final_status}<br>
        • Agents Used: {', '.join(agents_used)}<br>
        • Total Messages: {total_messages}
    </div>
    """, unsafe_allow_html = True)
    
    # Display individual agent messages and check for reports
    messages = result.get('message_history', [])
    report_found = False
    
    for msg in messages:
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('agent'):
            agent = msg.additional_kwargs['agent']
            content = msg.content
            
            if agent == "coordinator":
                display_message(content, "coordinator")
            elif agent == "retrieval":
                display_message(content, "retrieval")
            elif agent == "report":
                # Check if this is a report with JSON data
                report_data = parse_report_data(content)
                if report_data:
                    display_message("Report generated successfully! See dashboard below.", "report")
                    st.markdown("---")
                    display_report_dashboard(report_data)
                    report_found = True
                else:
                    display_message(content, "report")
    
    return report_found

def display_example_queries():
    """Display example queries in the sidebar"""
    st.sidebar.markdown("### Example Queries")
    
    database_examples = [
        "Find all defective headphones",
        "Add order 9999 with broken tablet costing $400",
        "Generate a report on expensive returns over $500",
        "Show me all approved camera returns",
        "What are the top returned products?",
        "Create a summary report for January 2025",
    ]
    
    non_database_examples = [
        "Hello, how are you?",
        "What's the weather like?",
    ]
    
    st.sidebar.markdown("**Database Queries:**")
    for example in database_examples:
        st.sidebar.markdown(f"• {example}")
    
    st.sidebar.markdown("")
    st.sidebar.markdown("**Non-Database Queries:**")
    for example in non_database_examples:
        st.sidebar.markdown(f"• {example}")
    
    st.sidebar.markdown("")
    st.sidebar.info("Copy any example above and paste it into the chat input field.")

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Auto-initialize coordinator on startup
    if not st.session_state.coordinator and not st.session_state.initialization_attempted:
        st.session_state.initialization_attempted = True
        with st.spinner("Initializing system..."):
            st.session_state.coordinator = setup_coordinator()
        
        if st.session_state.coordinator:
            st.success("System ready! You can start submitting queries.")
        else:
            st.error("System initialization failed. Please check your configuration.")
            st.info("You can try reinitializing using the sidebar button.")
    
    # Header
    st.markdown('<h1 class="main-header">MCP Database Coordinator</h1>', unsafe_allow_html = True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("System Control")
    
    # System status
    if st.sidebar.button("Reinitialize System (Optional)"):
        # Create progress indicators
        init_status = st.sidebar.empty()
        init_progress = st.sidebar.empty()
        
        try:
            # Step 1: Check environment
            with init_status.container():
                st.info("Checking environment...")
            progress_bar = init_progress.progress(0)
            time.sleep(0.5)
            
            # Step 2: Load API key
            progress_bar.progress(25)
            with init_status.container():
                st.info("Loading API key...")
            time.sleep(0.5)
            
            # Step 3: Initialize coordinator
            progress_bar.progress(50)
            with init_status.container():
                st.info("Initializing coordinator...")
            time.sleep(0.5)
            
            # Step 4: Setup agents
            progress_bar.progress(75)
            with init_status.container():
                st.info("Setting up agents...")
            
            # Actual initialization
            st.session_state.coordinator = setup_coordinator()
            st.session_state.initialization_attempted = True  # Mark as attempted
            
            # Complete
            progress_bar.progress(100)
            with init_status.container():
                if st.session_state.coordinator:
                    st.success("System ready!")
                else:
                    st.error("Initialization failed!")
            time.sleep(1)
            
            # Clear indicators
            init_status.empty()
            init_progress.empty()
            
        except Exception as e:
            init_status.empty()
            init_progress.empty()
            st.sidebar.error(f"Error: {str(e)}")
        
        st.rerun()
    
    # Display system status
    if st.session_state.coordinator:
        st.sidebar.success("Coordinator Ready")
    else:
        st.sidebar.error("Coordinator Failed to Initialize")
        st.sidebar.info("Use the button above to reinitialize the system")
    
    # Display example queries
    display_example_queries()
    
    # Clear chat history
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Database info
    st.sidebar.markdown("### 📊 Database Schema")
    st.sidebar.markdown("""
    **Fields:**
    - order_id (integer)
    - product (text)
    - category (text) 
    - return_reason (text)
    - cost (numeric)
    - approved_flag (Yes/No)
    - store_name (text)
    - date (YYYY-MM-DD)
    """)
    
    # Main chat interface
    with st.container():
        st.markdown("### Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message["content"], message["type"])
        
        # Add some spacing before input
        st.markdown("<br>", unsafe_allow_html = True)
        
        # Initialize input clearing mechanism
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False
        if 'input_counter' not in st.session_state:
            st.session_state.input_counter = 0
        
        # Query input with dynamic key to force clearing
        input_key = f"query_input_{st.session_state.input_counter}"
        query_input = st.text_input(
            "Enter your query:",
            key = input_key,
            value = st.session_state.get('current_query', ''),
            placeholder = "e.g., Find all defective headphones..."
        )
        
        # Clear the current query after using it
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        # Initialize submitted flag
        if 'query_submitted' not in st.session_state:
            st.session_state.query_submitted = False
        
        # Submit button
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            submit_clicked = st.button("Submit", type = "primary")
        with col_btn2:
            if st.button("Try Example"):
                st.session_state.current_query = "Find all defective headphones"
                st.rerun()
        
        # Process query
        if submit_clicked and query_input and not st.session_state.query_submitted:
            # Set submitted flag to prevent reprocessing
            st.session_state.query_submitted = True
            
            if not st.session_state.coordinator:
                st.error("System not initialized. Please use the reinitialize button in the sidebar.")
                st.session_state.query_submitted = False  # Reset flag on error
            else:
                # Show immediate confirmation
                st.info(f"**Submitted:** {query_input}")
                
                # Add user message immediately
                st.session_state.messages.append({
                    "content": query_input,
                    "type": "user",
                    "timestamp": datetime.now()
                })
                
                # Show immediate processing feedback
                with st.spinner("Processing your query..."):
                    try:
                        # Create status updates
                        status_placeholder = st.empty()
                        progress_placeholder = st.empty()
                        
                        # Show progress
                        with status_placeholder.container():
                            st.info("**Coordinator analyzing query...**")
                        progress_bar = progress_placeholder.progress(10)
                        
                        # Process the query
                        result = st.session_state.coordinator.process_request(query_input)
                        
                        # Update progress based on what happened
                        agents_used = result.get('agents_used', [])
                        
                        if 'coordinator' in agents_used:
                            progress_bar.progress(30)
                            with status_placeholder.container():
                                st.info("**Coordinator made routing decision...**")
                            time.sleep(0.3)
                        
                        if 'retrieval' in agents_used:
                            progress_bar.progress(60)
                            with status_placeholder.container():
                                st.info("**Retrieval agent processing data...**")
                            time.sleep(0.3)
                        
                        if 'report' in agents_used:
                            progress_bar.progress(80)
                            with status_placeholder.container():
                                st.info("**Report agent generating analysis...**")
                            time.sleep(0.3)
                        
                        # Complete
                        progress_bar.progress(100)
                        with status_placeholder.container():
                            st.success("**Processing complete!**")
                        time.sleep(0.5)
                        
                        # Clear progress indicators
                        status_placeholder.empty()
                        progress_placeholder.empty()
                        
                        # Add result to messages
                        st.session_state.messages.append({
                            "content": f"Coordinator processed your query",
                            "type": "system", 
                            "timestamp": datetime.now(),
                            "result": result
                        })
                        
                        st.success("Query processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.messages.append({
                            "content": f"Error processing query: {str(e)}",
                            "type": "error",
                            "timestamp": datetime.now()
                        })
                    
                    finally:
                        # Reset the submitted flag so user can submit new queries
                        st.session_state.query_submitted = False
                
                # Clear the input field by incrementing the counter (creates new widget)
                st.session_state.input_counter += 1
                
                # Force refresh to show all results
                st.rerun()
        
        elif submit_clicked and not query_input:
            st.error("Please enter a query first.")
        elif submit_clicked and st.session_state.query_submitted:
            pass  # Already processing
    
    # Display detailed results for the most recent query
    if st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        if "result" in latest_message:
            st.markdown("---")
            st.markdown("### Detailed Results")
            report_displayed = process_coordinator_result(latest_message["result"])
            
            # If no report was displayed, show a note about report generation
            if not report_displayed and any("report" in result.get('agents_used', []) for result in [latest_message.get("result", {})] if "result" in latest_message):
                st.info("Reports are automatically generated for database queries and displayed above.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        MCP Database Coordinator | Built with Streamlit & LangGraph
    </div>
    """, unsafe_allow_html = True)

if __name__ == "__main__":
    main()
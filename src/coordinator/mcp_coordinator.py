"""
I-Hsiu Kao
MCP-like Coordinator using LangGraph Supervisor Framework

This module implements a multi-agent coordination system that manages the retrieval agent and report agent using a supervisor pattern.
"""

import os
from typing import Dict, List, Optional, Any, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from agents.retrieval_agent import RetrievalAgent
from agents.report_agent import ReportAgent

# State management for the multi-agent system
class CoordinatorState(TypedDict):
    """State for agents in the coordinator system"""
    messages: Annotated[List[Any], add_messages]
    current_agent: Optional[str]
    task_completed: bool
    context_data: Dict[str, Any]
    user_query: str


class AgentCommand(BaseModel):
    """Response structure for routing between agents"""
    next_agent: str = Field(description = "Next agent to route to: 'retrieval', 'report', 'supervisor', or 'END'")
    reasoning: str = Field(description = "Explanation for routing decision")
    task_type: str = Field(description = "Type of task: 'data_retrieval', 'report_generation', 'analysis', 'coordination'")


class MCPCoordinator:
    """MCP-like Coordinator implemented with LangGraph workflow"""
    
    def __init__(self, openai_api_key: str, db_path: str = "output/data/demo.db", 
                 faiss_index_path: str = "output/data/demo.index", model: str = "gpt-4o-mini"):
        """Initialize the coordinator"""
        self.api_key = openai_api_key
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.model = model
        
        #initialize the coordinator LLM for initial routing decision
        self.coordinator_llm = ChatOpenAI(
            model = self.model,
            temperature = 0.1,
            api_key = openai_api_key
        )
        
        #initialize retrieval agent
        self.retrieval_agent = RetrievalAgent(
            openai_api_key = openai_api_key,
            db_path = db_path,
            faiss_index_path = faiss_index_path,
            model = self.model
        )
        #initialize report agent
        self.report_agent = ReportAgent(
            openai_api_key = openai_api_key,
            db_path = db_path,
            faiss_index_path = faiss_index_path,
            model = self.model
        )
        
        #build the graph
        self.graph = self._build_coordinator_graph()
    
    def _build_coordinator_graph(self) -> StateGraph:
        """Build a LangGraph supervisor flow with the coordinator and agents"""
        #create the state graph
        workflow = StateGraph(CoordinatorState)
        
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("report", self._report_node)
        
        workflow.add_edge(START, "coordinator")
        
        workflow.add_edge("retrieval", "report")
        workflow.add_edge("report", END)
        
        workflow.add_conditional_edges(
            "coordinator",
            self._coordinator_routing,
            {
                "retrieval": "retrieval",
                "END": END
            }
        )
        
        return workflow.compile()
    
    def _coordinator_node(self, state: CoordinatorState) -> CoordinatorState:
        """The coordinator node that decides if user input is database/query related"""
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])
        context_data = state.get("context_data", {})
        
        # Build coordinator prompt for database-related detection
        coordinator_prompt = f"""You are an intelligent routing coordinator for a database query system.

Your ONLY job is to determine if the user's input is related to querying, searching, adding, or analyzing database/CSV data.

The database contains return order data with these fields:
- order_id, product, category, return_reason, cost, approved_flag, store_name, date

DATABASE-RELATED queries include:
- Searching for records ("Find all defective headphones", "Show me returns over $500")
- Adding new records ("Add order 123 with broken camera")  
- Generating reports ("Create a summary report", "Analyze returns by store")
- Data analysis ("What are the top returned products?")

NON-DATABASE queries include:
- General questions ("What is the weather?", "How are you?")
- Unrelated requests ("Write me a poem", "Explain quantum physics")
- Greetings without data intent ("Hello", "Thanks, goodbye")

USER INPUT: "{user_query}"

Respond with ONLY ONE WORD:
- "DATABASE" if the query is related to database operations, queries, or reports
- "END" if the query is unrelated to the database system

Your response:"""
        
        #get coordinator decision
        response = self.coordinator_llm.invoke([
            SystemMessage(content = coordinator_prompt),
            HumanMessage(content = user_query)
        ])
        
        #parse the response to extract routing decision
        coordinator_content = response.content.strip().upper()
        
        #determine routing based on response
        if "DATABASE" in coordinator_content:
            next_agent = "retrieval"
            reasoning = f"User query is database-related, routing to retrieval agent"
        else:
            next_agent = "END"
            reasoning = f"User query is not database-related, ending workflow"
        
        #add coordinator message to history
        coordinator_message = AIMessage(
            content = f"Coordinator Decision: {reasoning}",
            additional_kwargs = {"agent": "coordinator", "routing": next_agent}
        )
        
        return CoordinatorState(
            messages = state["messages"] + [coordinator_message],
            current_agent = next_agent,
            task_completed = (next_agent == "END"),
            context_data = context_data,
            user_query = user_query
        )
    
    def _coordinator_routing(self, state: CoordinatorState) -> str:
        """
        Determine routing based on coordinator decision.
        """
        current_agent = state.get("current_agent")
        return current_agent if current_agent is not None else "END"
    
    def _retrieval_node(self, state: CoordinatorState) -> CoordinatorState:
        """Retrieval agent node that handles data operations"""
        user_query = state.get("user_query", "")
        
        #process query with retrieval agent
        try:
            result = self.retrieval_agent.process_query(user_query)
            
            #create response message
            retrieval_message = AIMessage(
                content = f"Retrieval Agent Result:\n{result}",
                additional_kwargs = {"agent": "retrieval"}
            )
            
            #update context with retrieval results
            context_data = state.get("context_data", {})
            context_data["last_retrieval"] = result
            context_data["retrieval_completed"] = True
            
            #check if this was a write operation that returned orders list
            if "Successfully inserted record" in result and "CURRENT RETURNED ORDERS LIST" in result:
                context_data["write_operation_completed"] = True
                context_data["orders_list_available"] = True
            
            #check if this was a read operation that returned query items list
            elif "QUERY RESULTS - RETURNED ORDERS LIST" in result and "records found" in result:
                context_data["read_operation_completed"] = True
                context_data["query_items_available"] = True
            
        except Exception as e:
            retrieval_message = AIMessage(
                content = f"Retrieval Agent Error: {str(e)}",
                additional_kwargs = {"agent": "retrieval", "error": True}
            )
            context_data = state.get("context_data", {})
            context_data["retrieval_error"] = str(e)
        
        return CoordinatorState(
            messages = state["messages"] + [retrieval_message],
            current_agent = "retrieval",
            task_completed = False,
            context_data = context_data,
            user_query = user_query
        )
    
    def _report_node(self, state: CoordinatorState) -> CoordinatorState:
        """Report agent node that handles report generation"""
        user_query = state.get("user_query", "")
        context_data = state.get("context_data", {})
        
        #get retrieval data to pass to report agent
        retrieval_data = context_data.get("last_retrieval", "")
        
        #create report query based on retrieval data and user intent
        if retrieval_data:
            #extract what type of data was retrieved to create appropriate report title
            if "CURRENT RETURNED ORDERS LIST" in retrieval_data:
                report_query = f"Generate a comprehensive report titled 'Returns Analysis Report' based on this data: {retrieval_data}"
            elif "QUERY RESULTS - RETURNED ORDERS LIST" in retrieval_data:
                report_query = f"Generate a comprehensive report titled 'Query Results Report' based on this data: {retrieval_data}"
            else:
                report_query = f"Generate a comprehensive report titled 'Data Analysis Report' based on this data: {retrieval_data}"
        else:
            #fallback if no retrieval data available
            report_query = f"Generate a report based on the user's request: {user_query}"
        
        #process query with report agent
        try:
            result = self.report_agent.process_query(report_query)
            
            #create response message
            report_message = AIMessage(
                content = f"Report Agent Result:\n{result}",
                additional_kwargs = {"agent": "report"}
            )
            
            #update context with report results
            context_data = state.get("context_data", {})
            context_data["last_report"] = result
            context_data["report_completed"] = True
            
        except Exception as e:
            report_message = AIMessage(
                content = f"Report Agent Error: {str(e)}",
                additional_kwargs = {"agent": "report", "error": True}
            )
            context_data = state.get("context_data", {})
            context_data["report_error"] = str(e)
        
        #report completed, flow will continue to END via graph edges
        return CoordinatorState(
            messages = state["messages"] + [report_message],
            current_agent = "report",
            task_completed = True,
            context_data = context_data,
            user_query = user_query
        )
    
    
    def process_request(self, user_query: str) -> Dict[str, Any]:
        """Process a user request through the multi-agent coordinator system"""
        print(f"\nMCP Coordinator: Processing request...")
        print(f"User Query: {user_query}")
        print("=" * 70)
        
        #initialize state
        initial_state = CoordinatorState(
            messages = [HumanMessage(content = user_query)],
            current_agent = None,
            task_completed = False,
            context_data = {},
            user_query = user_query
        )
        
        #execute the coordinator graph
        try:
            final_state = self.graph.invoke(initial_state, {'recursion_limit': 25})
            
            #format results
            messages = final_state.get("messages", [])
            context_data = final_state.get("context_data", {})
            
            #extract key results
            results = {
                "success": True,
                "user_query": user_query,
                "total_messages": len(messages),
                "agents_used": list(set([msg.additional_kwargs.get("agent", "unknown") 
                                       for msg in messages 
                                       if hasattr(msg, 'additional_kwargs')])),
                "context_data": context_data,
                "final_status": "completed" if final_state.get("task_completed") else "partial",
                "message_history": messages
            }
            
            #print summary
            print("\nCoordinator Summary:")
            print(f"Task Status: {results['final_status']}")
            print(f"Agents Used: {', '.join(results['agents_used'])}")
            print(f"Total Messages: {results['total_messages']}")
            
            #print agent results
            for msg in messages:
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get("agent"):
                    agent_name = msg.additional_kwargs["agent"]
            return results
            
        except Exception as e:
            error_result = {
                "success": False,
                "user_query": user_query,
                "error": str(e),
                "message": f"Coordinator failed: {e}"
            }
            print(f"\nCoordinator Error: {e}")
            return error_result


def main():

    print("MCP Coordinator Demo - Hard-coded LangGraph Workflow")
    print("=" * 80)
    
    #get API key
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    #initialize coordinator
    coordinator = MCPCoordinator(
        openai_api_key = api_key,
        db_path = "output/data/demo.db",
        faiss_index_path = "output/data/demo.index",
        model = model_name
    )
    
    #test queries that require different agent combinations
    test_queries = [
        #pure retrieval tasks
        "Find all defective headphones in the database",
        
        #data first, then report
        "Generate a report on expensive rejected returns over $500",
        
        #complex multi-step task
        "I need to analyze return patterns for electronics category - show me the data first, then create a summary report",
        
        #simple report request
        "Create a monthly report for January 2025 returns"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        
        result = coordinator.process_request(query)
        
        if result["success"]:
            print(f"\nTest {i} Completed Successfully")
        else:
            print(f"\nTest {i} Failed: {result.get('error', 'Unknown error')}")
        
        print("-" * 80)


if __name__ == "__main__":
    main()

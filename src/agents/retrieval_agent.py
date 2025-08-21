import sqlite3
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from core.csv_embeddings_processor import CSVEmbeddingsProcessor
    
load_dotenv()

CAT_VALUES = ['product', 'category', 'return_reason', 'approved_flag', 'store_name']

current_db_path = "sample_data.db"
processor = None

def init_processor(api_key: str, db_path: str, index_path: str):
    """Initialize the global processor and set database path"""
    global processor, current_db_path
    current_db_path = db_path
    processor = CSVEmbeddingsProcessor(
        openai_api_key = api_key,
        db_path = db_path,
        faiss_index_path = index_path
    )

def resolve_categorical_values(search_query: str, column: str) -> str:
    """Use FAISS similarity search to resolve categorical values"""
    try:
        if processor is None:
            return search_query
            
        search_text = f"{column}: {search_query}"
        faiss_results = processor.similarity_search(search_text.lower(), k = 1)
        
        for category_text, score in faiss_results:
            if category_text.startswith(f"{column}:") and score > 0.7:
                return category_text.split(":", 1)[1].strip()
        
        return search_query
    except Exception as e:
        print(f"Error resolving '{search_query}' for {column}: {e}")
        return search_query

class WriteDataInput(BaseModel):
    """Input schema for writing data to the database"""
    order_id: int = Field(description="Order ID (required)")
    product: Optional[str] = Field(None, description="Product name")
    category: Optional[str] = Field(None, description="Product category") 
    return_reason: Optional[str] = Field(None, description="Reason for return")
    cost: Optional[float] = Field(None, description="Cost of the item")
    approved_flag: Optional[str] = Field(None, description="Approval status (Yes/No)")
    store_name: Optional[str] = Field(None, description="Store name")
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")


class ReadDataInput(BaseModel):
    """Input schema for reading data from the database."""
    order_id: Optional[int] = Field(None, description="Order ID to filter by")
    product: Optional[str] = Field(None, description="Product name to search for")
    category: Optional[str] = Field(None, description="Product category to filter by")
    return_reason: Optional[str] = Field(None, description="Return reason to search for")
    cost_min: Optional[float] = Field(None, description="Minimum cost filter")
    approved_flag: Optional[str] = Field(None, description="Approval status (Yes/No)")
    store_name: Optional[str] = Field(None, description="Store name to filter by")
    date_filter: Optional[str] = Field(None, description="Date filter (partial match)")


@tool
def write_data(data: WriteDataInput) -> str:
    """Write a single row of data to the database"""
    try:
        # prepare data for insertion
        insert_data = {"order_id": data.order_id}
        
        # resolve categorical values using FAISS
        if data.product:
            insert_data["product"] = resolve_categorical_values(data.product, "product")
        if data.category:
            insert_data["category"] = resolve_categorical_values(data.category, "category")
        if data.return_reason:
            insert_data["return_reason"] = resolve_categorical_values(data.return_reason, "return_reason")
        if data.approved_flag:
            insert_data["approved_flag"] = resolve_categorical_values(data.approved_flag, "approved_flag")
        if data.store_name:
            insert_data["store_name"] = resolve_categorical_values(data.store_name, "store_name")
        
        # add non-categorical fields directly
        if data.cost is not None:
            insert_data["cost"] = data.cost
        if data.date:
            insert_data["date"] = data.date
        
        # form INSERT query using f-strings
        columns = list(insert_data.keys())
        placeholders = ', '.join(['?' for _ in columns])
        sql = f"INSERT INTO csv_data ({', '.join(columns)}) VALUES ({placeholders})"
        values = [insert_data[col] for col in columns]
        
        # execute query
        conn = sqlite3.connect(current_db_path)
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        
        # after successful write, return current list of all returned orders
        cursor.execute("SELECT * FROM csv_data ORDER BY date DESC, order_id DESC")
        column_names = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        conn.close()
        
        # format as returned orders list
        result_text = f"Successfully inserted record for order_id {data.order_id}.\n\n"
        result_text += f"CURRENT RETURNED ORDERS LIST ({len(results)} total records)\n"
        result_text += f"Search filters: All records\n\n"
        
        result_text += f"{'ORDER_ID':<8} | {'PRODUCT':<15} | {'STORE_NAME':<18} | {'DATE':<12} | {'REASON':<18} | {'STATUS':<8} | {'COST':<6}\n"
        result_text += f"{'-'*8}-+-{'-'*15}-+-{'-'*18}-+-{'-'*12}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}\n"
        
        # show all results
        display_results = results[:20]
        for row in display_results:
            row_dict = dict(zip(column_names, row))
            order_id = str(row_dict.get('order_id', 'N/A'))
            product = str(row_dict.get('product', 'N/A'))[:15]
            store_name = str(row_dict.get('store_name', 'N/A'))[:18] 
            date = str(row_dict.get('date', 'N/A'))[:12]
            return_reason = str(row_dict.get('return_reason', 'N/A'))[:18]
            status = 'Approved' if row_dict.get('approved_flag') == 'Yes' else 'Rejected'
            cost = f"${row_dict.get('cost', 'N/A')}"
            
            result_text += f"{order_id:<8} | {product:<15} | {store_name:<18} | {date:<12} | {return_reason:<18} | {status:<8} | {cost:<6}\n"
        
        if len(results) > 20:
            result_text += f"\n... and {len(results) - 20} more records."
        
        return result_text
        
    except sqlite3.IntegrityError as e:
        return f"Cannot insert: Order ID {data.order_id} already exists or constraint violation: {e}"
    except Exception as e:
        return f"Error writing data: {str(e)}"


@tool  
def read_data(filters: ReadDataInput) -> str:
    """Read data from the database with specified filters"""
    try:
        where_conditions = []
        params = []
        resolved_filters = {}
        
        # handle categorical filters with FAISS resolution
        if filters.order_id is not None:
            where_conditions.append("order_id = ?")
            params.append(filters.order_id)
            resolved_filters["order_id"] = filters.order_id
            
        if filters.product:
            resolved_product = resolve_categorical_values(filters.product, "product")
            where_conditions.append("product = ?")
            params.append(resolved_product)
            resolved_filters["product"] = resolved_product
            
        if filters.category:
            resolved_category = resolve_categorical_values(filters.category, "category")
            where_conditions.append("category = ?")
            params.append(resolved_category)
            resolved_filters["category"] = resolved_category
            
        if filters.return_reason:
            resolved_reason = resolve_categorical_values(filters.return_reason, "return_reason")
            where_conditions.append("return_reason = ?")
            params.append(resolved_reason)
            resolved_filters["return_reason"] = resolved_reason
            
        if filters.approved_flag:
            resolved_flag = resolve_categorical_values(filters.approved_flag, "approved_flag")
            where_conditions.append("approved_flag = ?")
            params.append(resolved_flag)
            resolved_filters["approved_flag"] = resolved_flag
            
        if filters.store_name:
            resolved_store = resolve_categorical_values(filters.store_name, "store_name")
            where_conditions.append("store_name = ?")
            params.append(resolved_store)
            resolved_filters["store_name"] = resolved_store
        
        # handle non-categorical filters directly
        if filters.cost_min is not None:
            where_conditions.append("cost >= ?")
            params.append(filters.cost_min)
            resolved_filters["cost_min"] = filters.cost_min
            
        if filters.date_filter:
            where_conditions.append("date LIKE ?")
            params.append(f"%{filters.date_filter}%")
            resolved_filters["date_filter"] = filters.date_filter
        
        if not where_conditions:
            return "Cannot search: Please provide at least one search filter."
        
        # form SELECT query using f-strings
        where_clause = " AND ".join(where_conditions)
        sql = f"SELECT * FROM csv_data WHERE {where_clause}"
        
        # execute query
        conn = sqlite3.connect(current_db_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        
        # get column names and results
        column_names = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No results found with filters: {resolved_filters}"
        
        # format results as a structured list of query items
        result_text = f"QUERY RESULTS - RETURNED ORDERS LIST ({len(results)} records found)\n"
        result_text += f"Search filters: {resolved_filters}\n\n"
        
        result_text += f"{'ORDER_ID':<8} | {'PRODUCT':<15} | {'STORE_NAME':<18} | {'DATE':<12} | {'REASON':<18} | {'STATUS':<8} | {'COST':<6}\n"
        result_text += f"{'-'*8}-+-{'-'*15}-+-{'-'*18}-+-{'-'*12}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}\n"
        
        # show all results
        display_results = results[:20]
        for row in display_results:
            row_dict = dict(zip(column_names, row))
            order_id = str(row_dict.get('order_id', 'N/A'))
            product = str(row_dict.get('product', 'N/A'))[:15]
            store_name = str(row_dict.get('store_name', 'N/A'))[:18] 
            date = str(row_dict.get('date', 'N/A'))[:12]
            return_reason = str(row_dict.get('return_reason', 'N/A'))[:18]
            status = 'Approved' if row_dict.get('approved_flag') == 'Yes' else 'Rejected'
            cost = f"${row_dict.get('cost', 'N/A')}"
            
            result_text += f"{order_id:<8} | {product:<15} | {store_name:<18} | {date:<12} | {return_reason:<18} | {status:<8} | {cost:<6}\n"
        
        if len(results) > 20:
            result_text += f"\n... and {len(results) - 20} more records. Use more specific filters to see fewer results."
        
        return result_text
        
    except Exception as e:
        return f"Error reading data: {str(e)}"


class RetrievalAgent:
    """LangChain-based AI retrieval agent using gpt-4o-mini and proper tool calling"""
    
    def __init__(self, openai_api_key: str, db_path: str = "sample_data.db", 
                 faiss_index_path: str = "sample_embeddings.index", model: str = "gpt-4o-mini"):
        """Initialize the LangChain AI-powered retrieval agent"""
        # initialize processor for FAISS similarity search
        init_processor(openai_api_key, db_path, faiss_index_path)
        
        # initialize ChatOpenAI model with gpt-4o-mini
        self.llm = ChatOpenAI(
            model = model,  # Latest mini model
            temperature = 0.1,
            api_key = openai_api_key
        )
        
        # bind tools to the model
        self.llm_with_tools = self.llm.bind_tools([write_data, read_data])
        
        # system prompt for the AI agent
        self.system_prompt = """You are an intelligent database assistant that helps users interact with a CSV database using natural language.

The database contains these columns:
- order_id (integer, required for writes)  
- product (categorical: Headphones, Keyboard, Camera, Tablet, etc.)
- category (categorical: Electronics, Accessories)
- return_reason (categorical: Defective, Warranty Claim, Not Compatible, etc.) 
- cost (numeric)
- approved_flag (categorical: Yes, No)
- store_name (categorical: various store names)
- date (text format: YYYY-MM-DD)

You have access to two tools:
1. write_data: Insert new records (requires order_id)
2. read_data: Search and retrieve records (requires at least one filter)

Instructions:
- For WRITE operations: Use write_data tool with order_id and other fields
- For READ operations: Use read_data tool with appropriate filters  
- Categorical values will be automatically resolved using similarity search
- Always use the tools rather than generating SQL directly
- Be helpful and provide clear explanations of what you're doing

Examples:
- "Add order 123 with defective headphones" → use write_data
- "Find all broken cameras" → use read_data with product="camera", return_reason="broken"
- "Show approved returns from January" → use read_data with approved_flag="yes", date_filter="2025-01"
"""
    
    def process_query(self, user_query: str) -> str:
        """Process a natural language query using LangChain tool calling"""
        try:
            # create messages
            messages = [
                SystemMessage(content = self.system_prompt),
                HumanMessage(content = user_query)
            ]
            
            # invoke the model with tools
            response = self.llm_with_tools.invoke(messages)
            
            # check if the model wants to call a tool
            if response.tool_calls:
                # Execute the first tool call
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"AI decided to use tool: {tool_name}")
                print(f"Tool arguments: {tool_args}")
                
                # execute the tool based on the name
                if tool_name == "write_data":
                    # extract the nested data if it exists
                    if "data" in tool_args:
                        write_input = WriteDataInput(**tool_args["data"])
                    else:
                        write_input = WriteDataInput(**tool_args)
                    result = write_data.invoke({"data": write_input})
                elif tool_name == "read_data":
                    # extract the nested filters if it exists
                    if "filters" in tool_args:
                        read_input = ReadDataInput(**tool_args["filters"])
                    else:
                        read_input = ReadDataInput(**tool_args)
                    result = read_data.invoke({"filters": read_input})
                else:
                    result = f"Unknown tool: {tool_name}"
                
                return result
            else:
                return f"{response.content}"
                
        except Exception as e:
            return f"Error processing query: {str(e)}"


def main():
    """Test the LangChain-based AI retrieval agent"""
    
    print("=== LangChain AI Retrieval Agent (gpt-4o-mini) ===\n")
    
    # get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your .env file or environment variables")
        return
    
    # initialize agent
    agent = RetrievalAgent(
        openai_api_key = api_key,
        db_path = "sample_data.db", 
        faiss_index_path = "sample_embeddings.index",
        model = "gpt-4o-mini"
    )
    
    # test queries
    test_queries = [
        # write operations
        "Add order 9001 with defective headphones from harbor point store",
        "Insert order 9002 - tablet with broken screen, cost $350, approved",
        
        # read operations
        "Find all defective headphones",
        "Show me approved camera returns",
        "Get all orders over $500",
        "Find returns from January 2025",
        
        # complex queries
        "What bluetooth speakers had battery problems?",
        "Show rejected warranty claims from electronics stores"
    ]
    
    for query in test_queries:
        print(f"User: '{query}'")
        result = agent.process_query(query)
        print(f"Agent: {result}\n")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()

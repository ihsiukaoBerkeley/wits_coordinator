import sqlite3
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
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


class ReportFiltersInput(BaseModel):
    """Input schema for filtering data for reports"""
    order_id: Optional[int] = Field(None, description="Order ID to filter by")
    product: Optional[str] = Field(None, description="Product name to search for")
    category: Optional[str] = Field(None, description="Product category to filter by")
    return_reason: Optional[str] = Field(None, description="Return reason to search for")
    cost_min: Optional[float] = Field(None, description="Minimum cost filter")
    cost_max: Optional[float] = Field(None, description="Maximum cost filter")
    approved_flag: Optional[str] = Field(None, description="Approval status (Yes/No)")
    store_name: Optional[str] = Field(None, description="Store name to filter by")
    date_filter: Optional[str] = Field(None, description="Date filter (partial match like '2025-01')")


class SummaryReportInput(BaseModel):
    """Input schema for generating summary reports"""
    title: str = Field(description="Report title")
    filters: Optional[ReportFiltersInput] = Field(None, description="Data filters to apply")
    group_by: Optional[str] = Field(None, description="Column to group data by (product, store_name, return_reason, etc.)")
    include_charts: Optional[bool] = Field(True, description="Whether to include visual charts")


@tool
def generate_web_report(report_input: SummaryReportInput) -> str:
    """Generate a structured web report with data analysis and findings"""
    try:
        import json
        
        # query data with filters
        where_conditions = []
        params = []
        resolved_filters = {}
        
        if report_input.filters:
            filters = report_input.filters
            
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
            
            # handle non-categorical filters
            if filters.cost_min is not None:
                where_conditions.append("cost >= ?")
                params.append(filters.cost_min)
                resolved_filters["cost_min"] = filters.cost_min
                
            if filters.cost_max is not None:
                where_conditions.append("cost <= ?")
                params.append(filters.cost_max)
                resolved_filters["cost_max"] = filters.cost_max
                
            if filters.date_filter:
                where_conditions.append("date LIKE ?")
                params.append(f"%{filters.date_filter}%")
                resolved_filters["date_filter"] = filters.date_filter
        
        # build query
        if where_conditions:
            where_clause = " WHERE " + " AND ".join(where_conditions)
        else:
            where_clause = ""
            
        sql = f"SELECT * FROM csv_data{where_clause}"
        
        # execute query
        conn = sqlite3.connect(current_db_path)
        df = pd.read_sql_query(sql, conn, params = params)
        conn.close()
        
        if df.empty:
            return "WEB_REPORT_JSON::{\"error\": \"No data found with the specified filters. Cannot generate report.\"}"
        
        # calculate key metrics
        total_cost = float(df['cost'].sum()) if 'cost' in df.columns else 0
        avg_cost = float(df['cost'].mean()) if 'cost' in df.columns else 0
        approval_rate = float((df['approved_flag'] == 'Yes').sum() / len(df) * 100) if 'approved_flag' in df.columns else 0
        
        # generate summary statistics
        summary_stats = {
            "total_records": len(df),
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "approval_rate": approval_rate
        }
        
        # top categories analysis
        top_analysis = {}
        if 'product' in df.columns:
            top_analysis["products"] = df['product'].value_counts().head(5).to_dict()
        if 'return_reason' in df.columns:
            top_analysis["return_reasons"] = df['return_reason'].value_counts().head(5).to_dict()
        if 'store_name' in df.columns:
            top_analysis["stores"] = df['store_name'].value_counts().head(5).to_dict()
        if 'category' in df.columns:
            top_analysis["categories"] = df['category'].value_counts().head(5).to_dict()
        
        # group by analysis
        group_analysis = None
        if report_input.group_by and report_input.group_by in df.columns:
            grouped = df.groupby(report_input.group_by).agg({
                'order_id': 'count',
                'cost': ['sum', 'mean'] if 'cost' in df.columns else 'count',
                'approved_flag': lambda x: (x == 'Yes').sum() / len(x) * 100 if 'approved_flag' in df.columns else 0
            }).round(2)
            
            # convert to dictionary format
            group_data = []
            for idx, row in grouped.iterrows():
                if 'cost' in df.columns:
                    group_data.append({
                        "group": str(idx),
                        "return_count": int(row[('order_id', 'count')]),
                        "total_cost": float(row[('cost', 'sum')]),
                        "avg_cost": float(row[('cost', 'mean')]),
                        "approval_rate": float(row[('approved_flag', '<lambda>')])
                    })
                else:
                    group_data.append({
                        "group": str(idx),
                        "return_count": int(row[('order_id', 'count')]),
                        "approval_rate": float(row[('approved_flag', '<lambda>')])
                    })
            
            group_analysis = {
                "group_by": report_input.group_by,
                "data": group_data
            }
        
        # convert DataFrame to records for raw data
        raw_data = df.to_dict('records')
        # convert numpy types to Python types for JSON serialization
        for record in raw_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    record[key] = str(value)
                elif hasattr(value, 'item'):  # numpy types
                    record[key] = value.item()
        
        # create comprehensive report structure
        report_data = {
            "title": report_input.title,
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filters_applied": resolved_filters,
            "summary_stats": summary_stats,
            "top_analysis": top_analysis,
            "group_analysis": group_analysis,
            "raw_data": raw_data[:100],
            "total_raw_records": len(raw_data)
        }
        
        # return JSON with special prefix for frontend parsing
        return f"WEB_REPORT_JSON::{json.dumps(report_data, default = str)}"
        
    except Exception as e:
        error_data = {"error": f"Error generating report: {str(e)}"}
        return f"WEB_REPORT_JSON::{json.dumps(error_data)}"


class ReportAgent:
    """Report Agent for generating Excel reports"""
    
    def __init__(self, openai_api_key: str, db_path: str = "sample_data.db", 
                 faiss_index_path: str = "sample_embeddings.index", model: str = "gpt-4o-mini"):
        """Initialize the LangChain Report Agent"""
        # initialize processor for FAISS similarity search
        init_processor(openai_api_key, db_path, faiss_index_path)
        
        # initialize ChatOpenAI model with gpt-4o-mini
        self.llm = ChatOpenAI(
            model = model,
            temperature = 0.1,
            api_key = openai_api_key
        )
        
        # bind tools to the model
        self.llm_with_tools = self.llm.bind_tools([generate_web_report])
        
        # system prompt for the Report Agent
        self.system_prompt = """You are an intelligent report generation assistant that creates web-based reports from return order data.

The database contains these columns:
- order_id (integer)
- product (categorical: Headphones, Keyboard, Camera, Tablet, etc.)
- category (categorical: Electronics, Accessories)  
- return_reason (categorical: Defective, Warranty Claim, Not Compatible, etc.)
- cost (numeric)
- approved_flag (categorical: Yes, No)
- store_name (categorical: various store names)
- date (text format: YYYY-MM-DD)

You have access to the generate_web_report tool which creates structured web reports with:
- Summary statistics with key metrics and findings
- Top analysis by categories (products, reasons, stores)
- Raw data tables for detailed viewing
- Grouped analysis by specified columns
- JSON data format for web display

Instructions:
- Always use the generate_web_report tool for report requests
- Extract filters from user queries (product, store, date range, etc.)
- Suggest meaningful grouping for analysis (by product, store, reason, etc.)
- Create descriptive report titles
- Handle both specific and broad reporting requests

Examples:
- "Generate a report on defective headphones" → filter by product and return_reason
- "Create a monthly report for January" → filter by date, group by various dimensions
- "Analyze returns by store performance" → group by store_name
- "Show me expensive rejected returns" → filter by cost and approval status
"""
    
    def process_query(self, user_query: str) -> str:
        """Process a natural language query for report generation"""
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
                # execute the first tool call
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"AI decided to generate report with tool: {tool_name}")
                print(f"Tool arguments: {tool_args}")
                
                # execute the tool
                if tool_name == "generate_web_report":
                    # extract the nested report_input if it exists
                    if "report_input" in tool_args:
                        report_input = SummaryReportInput(**tool_args["report_input"])
                    else:
                        report_input = SummaryReportInput(**tool_args)
                    result = generate_web_report.invoke({"report_input": report_input})
                else:
                    result = f"Unknown tool: {tool_name}"
                
                return result
            else:
                # no tool call, return the model's response
                return f"AI: {response.content}"
                
        except Exception as e:
            return f"Error processing report query: {str(e)}"


def main():
    """Test the LangChain-based Report Agent"""
    print("=== LangChain Report Agent (gpt-4o-mini) ===\n")
    
    # get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your .env file or environment variables")
        return
    
    # initialize agent
    agent = ReportAgent(
        openai_api_key = api_key,
        db_path = "sample_data.db",
        faiss_index_path = "sample_embeddings.index",
        model = "gpt-4o-mini"
    )
    
    # test report queries
    test_queries = [
        "Generate a summary report on all defective headphones",
        "Create a monthly analysis report for January 2025",
        "Generate a store performance report grouped by store name",
        "Create a report on expensive rejected returns over $500",
        "Generate a product category analysis report with charts"
    ]
    
    for query in test_queries:
        print(f"User: '{query}'")
        result = agent.process_query(query)
        print(f"Report Agent: {result}\n")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()

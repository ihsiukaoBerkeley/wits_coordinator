# MCP-like Coordinator

## System Architecture

User Query -> MCP Coordinator -> Route to Agent -> Retrieval -> Report Agent

The coordinator routes requests between two agents:
- **Database queries**: Retrieval Agent -> Report Agent -> Interactive Dashboard
- **Non-database queries**: Direct response

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── retrieval_agent.py    # Retrieval agent
│   │   └── report_agent.py       # Report agent
│   ├── coordinator/
│   │   └── mcp_coordinator.py    # Coordinator
│   └── core/  
│       ├── csv_embeddings_processor.py  # Create DB and indexes
│       └── sample.csv               # Sample data
├── streamlit_app.py         # Web interface
├── requirements.txt         # Dependencies
└── README.md
```

## Query Guidelines

To get the best results from the system, follow these guidelines when writing queries:

### **Database Operations**
The system recognizes these types of queries and routes them to the database agents:

**Search/Retrieval:**
- "Find all defective headphones"
- "Show me returns over $500"
- "Get approved camera returns from Best Buy"
- "Find warranty claims from January 2025"

**Adding New Records:**
- "Add order 9999 with broken tablet costing $400"
- "Insert return for order 1234: defective keyboard from Tech Store"
- "Add new record: order 5678, camera lens issue, $250, approved"

**Report Generation:**
- "Generate a report on expensive returns"
- "Create a summary for electronics category"
- "Analyze return patterns by store"
- "Show top returned products this month"

### **Non-Database Queries**
These queries will receive direct responses (not processed through database agents):

- General greetings: "Hello", "How are you?"
- Unrelated questions: "What's the weather?", "Tell me a joke"
- System help: "How does this work?"

### **Best Practices**

**For Search Queries:**
- Be specific about what you're looking for
- Use product names, store names, or reasons as they appear in the data
- The system uses fuzzy matching for categorical values
- Examples: "headphones" will match "Bluetooth Headphones"

**For Adding Records:**
- Always include an order ID (required field)
- Provide as much detail as possible
- Use natural language - the system will extract the fields
- Example: "Add order 1001 with defective mouse from Electronics Plus, cost $25, rejected"

**For Reports:**
- Specify what type of analysis you want
- Mention grouping preferences (by store, product, category, etc.)
- Use time ranges when needed
- Example: "Generate a store performance report grouped by location"

### **Available Data Fields**
When writing queries, you can reference these data fields:
- **Products**: Headphones, Keyboard, Camera, Tablet, Mouse, etc.
- **Categories**: Electronics, Accessories
- **Return Reasons**: Defective, Warranty Claim, Not Compatible, etc.
- **Stores**: Various store names (Best Buy, Tech Store, etc.)
- **Approval Status**: Yes/No (use "approved" or "rejected")
- **Dates**: Use YYYY-MM-DD format or partial matches like "January 2025"

### **Tips for Better Results**
- Use complete sentences rather than keywords
- Be specific about what you want to see in reports
- If searching fails, try different wording or partial matches
- The system learns from your query patterns and improves over time

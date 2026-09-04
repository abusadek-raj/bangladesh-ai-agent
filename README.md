
# Multi-Tool AI Agent for Bangladesh

A LangChain-powered multi-tool AI Agent designed to answer structured database queries about Bangladesh's institutions, hospitals, and restaurants, as well as fetch general context via real-time web search.

## Features
- **Institutions DB Tool:** Queries SQLite database containing university and institutional data.
- **Hospitals DB Tool:** Queries medical facilities, locations, and bed counts across Bangladesh.
- **Restaurants DB Tool:** Searches restaurant offerings, locations, and cuisines.
- **Web Search Tool:** Leverages Tavily API for general knowledge, healthcare policies, and cultural context.
- **LLM Backbone:** Powered by Google Gemini API (`gemini-3.5-flash-lite`).

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/bangladesh-ai-agent.git](https://github.com/your-username/bangladesh-ai-agent.git)
   cd bangladesh-ai-agent
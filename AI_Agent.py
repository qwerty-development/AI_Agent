from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from supabase import create_client, Client
from typing import cast
from datetime import datetime, timedelta
from dateutil import tz
# from availability_tools import (
#     check_any_time_slots as av_check_any_time_slots,
#     get_available_time_slots as av_get_available_time_slots,
#     get_table_options_for_slot as av_get_table_options_for_slot,
#     search_time_range as av_search_time_range,
# )
import json
from typing import List

load_dotenv()
url: str = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

# Create Supabase client if env vars are present; otherwise defer errors to tools
supabase: Optional[Client] = None
try:
    if url and key:
        supabase = create_client(url, key)
    else:
        print("Supabase env vars missing; tools will respond gracefully")
except Exception as _e:
    print(f"Failed to init Supabase client: {_e}")
    supabase = None

class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Get timezone for consistent date handling
_LOCAL_TZ = tz.gettz(os.getenv("AVAILABILITY_TZ", "Asia/Beirut")) or tz.UTC

def get_supabase_client() -> Optional[Client]:
    """Get the appropriate Supabase client (authenticated if available, otherwise global)"""
    import threading
    if hasattr(threading.current_thread(), 'supabase_client'):
        return threading.current_thread().supabase_client
    return supabase

# Conversation memory system
class ConversationMemory:
    def __init__(self, max_history: int = 6):
        """Initialize conversation memory with a maximum history limit.
        Reduced from 20 to 6 messages to optimize token usage and reduce costs.
        6 messages = 3 turns of conversation (user + assistant pairs), which is sufficient for context.
        """
        self.messages: List[BaseMessage] = []
        self.max_history = max_history
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.messages.append(message)
        # Keep only the most recent messages to prevent context overflow
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation history."""
        return self.messages.copy()
    
    def clear(self):
        """Clear the conversation history."""
        self.messages.clear()
    
    def get_context_size(self) -> int:
        """Get the number of messages in history."""
        return len(self.messages)

# Global conversation memory instance for interactive sessions
conversation_memory = ConversationMemory()

def create_conversation_memory(max_history: int = 6) -> ConversationMemory:
    """Create a new conversation memory instance for external use.
    Default reduced to 6 messages for cost optimization."""
    return ConversationMemory(max_history)

tools = []

# Note: finishedUsingTools tool removed - workflow now ends naturally when AI stops calling tools

@tool
def convertRelativeDate(relative_date: str) -> str:
    """Convert relative dates like 'today', 'tomorrow', 'yesterday' to YYYY-MM-DD format.
    Examples: 'today' -> '2025-08-14', 'tomorrow' -> '2025-08-15', 'next Monday' -> '2025-08-18'
    Always use this tool when user mentions relative dates before calling availability tools."""
    print(f"AI is converting relative date: {relative_date}")
    try:
        now_local = datetime.now(_LOCAL_TZ)
        today_local = now_local.date()
        
        relative_lower = relative_date.lower().strip()
        
        if relative_lower in ['today', 'tod', 'tonight']:
            return today_local.strftime("%Y-%m-%d")
        elif relative_lower in ['tomorrow', 'tmr', 'tmrw']:
            return (today_local + timedelta(days=1)).strftime("%Y-%m-%d")
        elif relative_lower in ['yesterday', 'yest']:
            return (today_local - timedelta(days=1)).strftime("%Y-%m-%d")
        elif relative_lower in ['day after tomorrow', 'overmorrow']:
            return (today_local + timedelta(days=2)).strftime("%Y-%m-%d")
        elif 'next week' in relative_lower:
            return (today_local + timedelta(days=7)).strftime("%Y-%m-%d")
        elif 'this weekend' in relative_lower:
            # Find next Saturday
            days_until_saturday = (5 - today_local.weekday()) % 7
            if days_until_saturday == 0 and now_local.hour > 18:  # If it's Saturday evening, go to next Saturday
                days_until_saturday = 7
            return (today_local + timedelta(days=days_until_saturday)).strftime("%Y-%m-%d")
        else:
            # If not a recognized relative date, return as-is (might be already in YYYY-MM-DD format)
            return relative_date
            
    except Exception as e:
        print(f"Error converting relative date: {e}")
        # Fallback to today if conversion fails
        return datetime.now(_LOCAL_TZ).date().strftime("%Y-%m-%d")

tools.append(convertRelativeDate)

system_prompt = """
You are DineMate, a specialized restaurant discovery assistant for Plate - a restaurant reservation platform.

## YOUR ROLE
Help users discover perfect restaurants by:
1. Understanding preferences (cuisine, atmosphere, special features)
2. Providing personalized recommendations using user profiles (allergies, dietary restrictions, favorite cuisines)
3. Explaining restaurant details (cuisine, price, ratings, features)
4. Answering general questions about dining and reservations
5. Maintaining a friendly, conversational tone

## AVAILABLE TOOLS
You have access to 7 specialized search tools:

1. **getAllRestaurants()** - Browse all restaurants (pre-sorted by featured status & ratings)
2. **getRestaurantsByCuisineType(cuisineType)** - Search by specific cuisine (e.g., "Italian", "Lebanese")
3. **getRestaurantsByName(query)** - Search by restaurant name or description
4. **searchRestaurantsByFeatures(outdoor_seating, shisha_available, min_rating, price_range)** - Filter by features
   - Use this for: "restaurants with shisha", "outdoor seating", "highly rated", "budget-friendly"
   - All parameters are optional, combine as needed
5. **searchRestaurantsByMenuItem(query)** - Find restaurants serving specific dishes
   - Use this for: "sushi", "pasta carbonara", "vegan burger", "chocolate cake", "tacos"
   - Searches menu item names and descriptions
6. **searchRestaurantsByMenuCategory(category_name)** - Find restaurants with specific menu categories
   - Use this for: "Desserts", "Appetizers", "Seafood", "Cocktails", "Vegetarian Options"
7. **convertRelativeDate(relative_date)** - Convert "today", "tomorrow" to dates

**Tool Selection Priority:**
1. Specific dish/food requests → **searchRestaurantsByMenuItem** (e.g., "sushi", "tiramisu", "tacos")
2. Menu category requests → **searchRestaurantsByMenuCategory** (e.g., "desserts", "appetizers")
3. Feature requests (shisha, outdoor, rating, price) → **searchRestaurantsByFeatures**
4. Cuisine requests → **getRestaurantsByCuisineType**
5. Name/description search → **getRestaurantsByName**
6. General browse → **getAllRestaurants**

**Tool Usage Rules:**
- Choose the MOST SPECIFIC tool for each query
- Call the appropriate search tool (getAllRestaurants, getRestaurantsByCuisineType, getRestaurantsByName, searchRestaurantsByFeatures, searchRestaurantsByMenuItem, searchRestaurantsByMenuCategory)
- After receiving results, respond with a friendly message
- Results are pre-sorted: featured restaurants appear first, then by rating
- You MUST extract IDs from tool results and include them in your response

## RESPONSE FORMAT - ABSOLUTELY MANDATORY
**EVERY restaurant recommendation MUST include this line:**
```
RESTAURANTS_TO_SHOW: id1,id2,id3,id4,id5
```

**Complete Example with ID Extraction:**

Tool returns: `[{"id": "a1b2c3", "name": "Sushi Bar", "rating": 4.5}, {"id": "d4e5f6", "name": "Pasta House", "rating": 4.8}]`

Your response:
```
I found amazing sushi restaurants! 🍣 Sushi Bar has a 4.5 rating, and Pasta House is highly rated at 4.8.
RESTAURANTS_TO_SHOW: a1b2c3,d4e5f6
```

**NON-NEGOTIABLE Rules:**
- If you used a search tool and got restaurant results, you MUST include RESTAURANTS_TO_SHOW
- Extract actual IDs from the JSON results (each restaurant object has an "id" field)
- Include up to 5 restaurant IDs separated by commas
- The line MUST start with exactly "RESTAURANTS_TO_SHOW: " followed by IDs
- Without this line, the frontend cannot display restaurants to users
- This is NOT optional - responses without IDs are considered failures

## PERSONALIZATION WITH USER PROFILES
When user profile data is provided (in context), apply these filters:

✅ **Must Consider:**
- **Allergies** - Exclude incompatible restaurants
- **Dietary Restrictions** - Filter by dietary_options field
- **Favorite Cuisines** - Prioritize preferred cuisine types

📊 **Use for Context:**
- **Preferred Party Size** - Default for availability queries
- **Loyalty Points** - Mention rewards/tier benefits if relevant

## WORKFLOW BY REQUEST TYPE

### 🔍 Restaurant Discovery/Recommendations:
**SIMPLE 3-STEP PROCESS:**

1. **Choose & call the right tool:**
   - Specific dish/food (sushi, pasta, tacos) → searchRestaurantsByMenuItem
   - Menu category (desserts, appetizers) → searchRestaurantsByMenuCategory
   - Features (shisha, outdoor, rating, price) → searchRestaurantsByFeatures
   - Cuisine (Italian, Lebanese) → getRestaurantsByCuisineType
   - Name/Description → getRestaurantsByName
   - General browse → getAllRestaurants

2. **Extract IDs from results:**
   Tool returns: `[{"id": "abc-123", "name": "Sushi Bar"}, {"id": "def-456", "name": "Pasta House"}]`
   Extract: `["abc-123", "def-456"]`

3. **Write response with IDs:**
   ```
   I found amazing sushi restaurants! 🍣
   RESTAURANTS_TO_SHOW: abc-123,def-456,ghi-789
   ```
   
**CRITICAL:** Always end with `RESTAURANTS_TO_SHOW:` line containing extracted IDs!

### 📅 Availability Questions:
**Note:** Availability tools are currently disabled. For availability queries:
1. Use convertRelativeDate for "today"/"tomorrow"
2. Find restaurant with getRestaurantsByName
3. Inform user to check restaurant's opening hours (from restaurant data)
4. Guide them to make a reservation through the app

### 🍽️ Cuisine Lists:
When asked "what cuisines do you have?":
- Use the provided "AVAILABLE CUISINE TYPES:" context
- DO NOT call tools
- Return concise, formatted list

## CONVERSATION GUIDELINES

**Stay Focused:**
- ONLY answer restaurant/dining questions
- Politely redirect off-topic requests: "I specialize in restaurant recommendations. How can I help you find a great place to eat?"

**Be Conversational:**
- Use natural language, not robotic responses
- Show enthusiasm about food and dining
- Ask clarifying questions when needed (party size, cuisine preference, occasion)

**Data Integrity:**
- Base ALL answers on real tool data
- Never fabricate restaurant information
- If no matches found, suggest alternatives

## IMPORTANT CONSTRAINTS
❌ **Never Do:**
- Provide code, technical solutions, or programming help
- Discuss topics outside restaurant/dining
- Make up restaurant IDs or details
- Call tools for cuisine lists (use provided context)

✅ **Always Do:**
- Use RESTAURANTS_TO_SHOW format for recommendations
- Extract IDs from tool results and include them
- Consider user profile data when available
- Select the most specific tool for each query
"""
restaurants_table_columns:str = "id, name, description, address, tags, opening_time, closing_time, cuisine_type, price_range, average_rating, dietary_options, ambiance_tags, outdoor_seating, shisha_available, ai_featured"

# User profile data is now pre-fetched and provided in conversation context
# No need for getUserProfile tool - data comes from external source for security

def fetch_user_profile(user_id: str, client: Optional[Client] = None) -> Optional[dict]:
    """Fetch user profile data externally (not as an AI tool).
    Returns user profile dict or None if not found/error."""
    try:
        client_to_use = client if client else supabase
        if not client_to_use or not user_id or not user_id.strip():
            return None
        
        print(f"Fetching user profile for user_id: {user_id}")
        result = (
            client_to_use
            .table("profiles")
            .select("full_name, allergies, favorite_cuisines, dietary_restrictions, preferred_party_size, loyalty_points")
            .eq("id", user_id.strip())
            .execute()
        )
        
        if not result.data or len(result.data) == 0:
            print(f"No profile found for user_id: {user_id}")
            return None
        
        user_profile = result.data[0]
        print(f"Retrieved user profile: {user_profile}")
        return user_profile
        
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

def getAllCuisineTypes() -> str:
    """Return the unique cuisine types available in the application as a JSON list string.
    Note: This is no longer exposed as an AI tool; it's used internally and by the API.
    """
    print("Fetching cuisine types (internal)")
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        result = client.table("restaurants").select("cuisine_type").execute()
        cuisineTypes = result.data

        if not cuisineTypes:
            return json.dumps([])

        # Extract unique cuisine types
        unique_cuisines = list(set([item['cuisine_type'] for item in cuisineTypes if item.get('cuisine_type')]))
        print(f"Found cuisine types: {unique_cuisines}")
        return json.dumps(unique_cuisines)
    except Exception as e:
        print(f"Error fetching cuisine types: {e}")
        return json.dumps([])

@tool
def getRestaurantsByCuisineType(cuisineType: str) -> str:
    """Request restaurants from the database based on the cuisine type"""
    cuisineType=cuisineType.strip().capitalize()
    print(f"AI is looking for restaurants with cuisine type: {cuisineType}")
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        # Use ilike for case-insensitive matching in PostgreSQL/Supabase with wildcards
        pattern = f"%{cuisineType}%" if cuisineType else "%"
        result = (
            client
            .table("restaurants")
            .select(restaurants_table_columns)
            .ilike("cuisine_type", pattern)
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .execute()
        )
        restaurants = result.data
        
        if not restaurants:
            return f"No restaurants found with cuisine type: {cuisineType}"
        
        print(f"Found {len(restaurants)} restaurants")
        return json.dumps(restaurants)
    except Exception as e:
        print(f"Error fetching restaurants: {e}")
        return f"Error retrieving restaurants for cuisine type: {cuisineType}"

tools.append(getRestaurantsByCuisineType)

@tool
def getAllRestaurants() -> str:
    """Request all restaurants with all their info from the database. Returns top 20 restaurants sorted by featured status and rating."""
    print("AI is looking for all restaurants")
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        result = (
            client
            .table("restaurants")
            .select(restaurants_table_columns)
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .eq("status", "active")
            .limit(100)
            .execute()
        )
        restaurants = result.data

        if not restaurants:
            return "No restaurants found"
        print(f"Found {len(restaurants)} restaurants")
        return json.dumps(restaurants)
    
    except Exception as e:
        print(f"Error fetching restaurants: {e}")
        return "Error retrieving restaurants"

tools.append(getAllRestaurants)


@tool
def getRestaurantsByName(query: str) -> str:
    """Search restaurants by name or description. Case-insensitive, partial matches supported."""
    q = (query or "").strip()
    print(f"AI is searching restaurants by name/description: {q}")
    try:
        if not supabase:
            return json.dumps([])
        pattern = f"%{q}%" if q else "%"
        result = (
            supabase
            .table("restaurants")
            .select(restaurants_table_columns)
            .ilike("name", pattern)
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .limit(50)
            .execute()
        )
        # If name search is too strict and empty, fallback to description search
        restaurants = result.data or []
        if not restaurants and q:
            result_desc = (
                supabase
                .table("restaurants")
                .select(restaurants_table_columns)
                .ilike("description", pattern)
                .order("ai_featured", desc=True)
                .order("average_rating", desc=True)
                .limit(50)
                .execute()
            )
            restaurants = result_desc.data or []
        return json.dumps(restaurants)
    except Exception as e:
        print(f"Error searching restaurants by name: {e}")
        return json.dumps([])

tools.append(getRestaurantsByName)

@tool
def searchRestaurantsByFeatures(outdoor_seating: Optional[bool] = None, shisha_available: Optional[bool] = None, min_rating: Optional[float] = None, price_range: Optional[int] = None) -> str:
    """Search restaurants by common features and filters.
    - outdoor_seating: True to find restaurants with outdoor seating
    - shisha_available: True to find restaurants that offer shisha
    - min_rating: Minimum average rating (e.g., 4.0)
    - price_range: Exact price level 1-4 (1=budget, 4=expensive)
    Returns JSON list of matching restaurants sorted by featured status and rating."""
    
    print(f"AI is searching by features: outdoor={outdoor_seating}, shisha={shisha_available}, rating>={min_rating}, price={price_range}")
    
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        
        query = client.table("restaurants").select(restaurants_table_columns)
        
        # Apply filters only if specified
        if outdoor_seating is not None:
            query = query.eq("outdoor_seating", outdoor_seating)
        
        if shisha_available is not None:
            query = query.eq("shisha_available", shisha_available)
        
        if min_rating is not None:
            query = query.gte("average_rating", float(min_rating))
        
        if price_range is not None:
            query = query.eq("price_range", int(price_range))
        
        result = (
            query
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .limit(20)
            .execute()
        )
        
        restaurants = result.data
        
        if not restaurants:
            return json.dumps([])
        
        print(f"Found {len(restaurants)} restaurants matching features")
        return json.dumps(restaurants)
        
    except Exception as e:
        print(f"Error searching by features: {e}")
        return json.dumps([])

tools.append(searchRestaurantsByFeatures)

@tool
def searchRestaurantsByMenuItem(query: str) -> str:
    """Search restaurants that serve specific menu items or dishes.
    Searches through menu item names and descriptions.
    Examples: "sushi", "pasta carbonara", "chocolate cake", "tacos", "vegan burger"
    Returns JSON list of restaurants that have matching menu items."""
    
    q = (query or "").strip()
    print(f"AI is searching restaurants by menu item: {q}")
    
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        
        if not q:
            return json.dumps([])
        
        pattern = f"%{q}%"
        
        # Get restaurant IDs that have matching menu items (name or description)
        menu_result = (
            client
            .table("menu_items")
            .select("restaurant_id")
            .or_(f"name.ilike.{pattern},description.ilike.{pattern}")
            .eq("is_available", True)
            .execute()
        )
        
        restaurant_ids = list(set([item['restaurant_id'] for item in menu_result.data if item.get('restaurant_id')]))
        
        if not restaurant_ids:
            print(f"No menu items found matching '{q}'")
            return json.dumps([])
        
        # Fetch full restaurant details
        restaurants_result = (
            client
            .table("restaurants")
            .select(restaurants_table_columns)
            .in_("id", restaurant_ids)
            .eq("status", "active")
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .limit(20)
            .execute()
        )
        
        print(f"Found {len(restaurants_result.data)} restaurants with menu item matching '{q}'")
        return json.dumps(restaurants_result.data)
        
    except Exception as e:
        print(f"Error searching by menu item: {e}")
        return json.dumps([])

tools.append(searchRestaurantsByMenuItem)

@tool
def searchRestaurantsByMenuCategory(category_name: str) -> str:
    """Search restaurants that offer specific menu categories.
    Examples: "Desserts", "Appetizers", "Main Courses", "Beverages", "Cocktails", "Seafood"
    Returns JSON list of restaurants that have the specified category."""
    
    cat = (category_name or "").strip()
    print(f"AI is searching restaurants by menu category: {cat}")
    
    try:
        client = get_supabase_client()
        if not client:
            return json.dumps([])
        
        if not cat:
            return json.dumps([])
        
        pattern = f"%{cat}%"
        
        # Get restaurant IDs that have matching menu categories
        categories_result = (
            client
            .table("menu_categories")
            .select("restaurant_id")
            .ilike("name", pattern)
            .eq("is_active", True)
            .execute()
        )
        
        restaurant_ids = list(set([item['restaurant_id'] for item in categories_result.data if item.get('restaurant_id')]))
        
        if not restaurant_ids:
            print(f"No restaurants found with category '{cat}'")
            return json.dumps([])
        
        # Fetch full restaurant details
        restaurants_result = (
            client
            .table("restaurants")
            .select(restaurants_table_columns)
            .in_("id", restaurant_ids)
            .eq("status", "active")
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .limit(20)
            .execute()
        )
        
        print(f"Found {len(restaurants_result.data)} restaurants with category '{cat}'")
        return json.dumps(restaurants_result.data)
        
    except Exception as e:
        print(f"Error searching by menu category: {e}")
        return json.dumps([])

tools.append(searchRestaurantsByMenuCategory)


# -----------------------------
# Availability tools (backend service key based)
# -----------------------------

# @tool
# def checkAnyTimeSlots(restaurant_id: str, date: str, party_size: int, user_id: Optional[str] = None) -> str:
#     """Return {"available": bool} if at least one slot exists for the given date and party size."""
#     try:
#         available = av_check_any_time_slots(restaurant_id, date, int(party_size), user_id)
#         return json.dumps({"available": bool(available)})
#     except Exception as e:
#         return json.dumps({"error": str(e)})

# tools.append(checkAnyTimeSlots)

# @tool
# def getAvailableTimeSlots(restaurant_id: str, date: str, party_size: int, user_id: Optional[str] = None) -> str:
#     """Return a JSON list of {time: 'HH:MM', available: true} slots for the day."""
#     try:
#         slots = av_get_available_time_slots(restaurant_id, date, int(party_size), user_id)
#         return json.dumps(slots)
#     except Exception as e:
#         return json.dumps({"error": str(e)})

# tools.append(getAvailableTimeSlots)

# @tool
# def getTableOptionsForSlot(restaurant_id: str, date: str, time: str, party_size: int, user_id: Optional[str] = None) -> str:
#     """Return table options for a specific time slot, or null if none."""
#     try:
#         # Note: The underlying function may not use user_id, but we keep it for consistency
#         options = av_get_table_options_for_slot(restaurant_id, date, time, int(party_size))
#         return json.dumps(options)
#     except Exception as e:
#         return json.dumps({"error": str(e)})

# tools.append(getTableOptionsForSlot)

# @tool
# def searchTimeRange(restaurant_id: str, date: str, start_time: str, end_time: str, party_size: int, user_id: Optional[str] = None) -> str:
#     """Return best table options for all available slots within a time range on a given date."""
#     try:
#         results = av_search_time_range(restaurant_id, date, start_time, end_time, int(party_size), user_id)
#         return json.dumps(results)
#     except Exception as e:
#         return json.dumps({"error": str(e)})

# tools.append(searchTimeRange)

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)
llm = llm.bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    """Our agent node that processes messages and generates responses."""
    messages = state["messages"]

    # OPTIMIZATION: Only add system prompt if this is the first message in the conversation
    # LangGraph maintains state across iterations, so we don't need to resend it every time
    # This reduces token usage by ~1000 tokens per LLM call after the first one
    has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)

    if has_system_message:
        # System message already in conversation, don't add again
        full_messages = messages
    else:
        # First message in conversation, add system prompt
        full_messages = [SystemMessage(content=system_prompt)] + messages

    print(f"Sending {len(full_messages)} messages to LLM (system_prompt_included: {not has_system_message})")

    # Get response from the model
    response = llm.invoke(full_messages)
    
    # print(f"LLM response type: {type(response)}")
    # print(f"LLM response content: {response.content}")
    # print(f"LLM tool calls: {response.tool_calls}")
    
    # Return the updated state with the new message
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determine whether to continue with tools or end the conversation.
    Simple logic: if AI wants to call tools, continue. Otherwise, end."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', []) or []
        print(f"Tool calls found: {len(tool_calls)}")
        
        # If there are tool calls, continue to execute them
        if tool_calls:
            print("🔧 AI has tool calls - continuing to tools")
            for call in tool_calls:
                print(f"  - {call['name']}")
            return "continue"
        
        # No tool calls means AI is done and ready to respond
        if last_message.content:
            print("✅ AI ready with response - ending")
            return "end"
    
    print("🔄 Default case - continuing")
    return "continue"

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

def chat_with_bot(user_input: str, memory: Optional[ConversationMemory] = None, user_id: Optional[str] = None, authenticated_client: Optional[Client] = None, current_user: Optional[dict] = None) -> str:
    """
    Function to chat with the bot. Can use conversation memory for context.
    If memory is provided, maintains conversation history.
    If memory is None, operates in stateless mode (for API usage).
    If user_id is provided, pre-fetches user profile for personalization.
    If authenticated_client is provided, uses it for database operations with RLS.
    """
    try:
        # Early handle: user asks to list available cuisines -> answer directly, no LLM/tools
        intent_text_early = (user_input or "").lower()
        if "cuisine" in intent_text_early or "cuisines" in intent_text_early:
            list_markers = [
                "what are", "what's", "list", "available", "do you have", "which",
                "what kind", "types of cuisine", "cuisine types", "type of cuisine", "types"
            ]
            if any(m in intent_text_early for m in list_markers) and not any(
                k in intent_text_early for k in ["near", "find", "recommend", "suggest", "restaurant", "restaurants", "where"]
            ):
                try:
                    cuisines_raw = getAllCuisineTypes()
                    cuisines_list = json.loads(cuisines_raw) if cuisines_raw else []
                except Exception:
                    cuisines_list = []
                if isinstance(cuisines_list, list) and cuisines_list:
                    try:
                        cuisines_list = sorted(set(map(str, cuisines_list)), key=lambda x: x.lower())
                    except Exception:
                        cuisines_list = list(dict.fromkeys(map(str, cuisines_list)))
                    return "The available cuisine types are: " + ", ".join(cuisines_list) + "."
                return "I couldn't retrieve the cuisine types right now. Please try again shortly."
        # Use authenticated client if provided, otherwise fall back to global supabase client
        client_to_use = authenticated_client if authenticated_client else supabase
        
        # Store client in a thread-local variable so tools can access it
        import threading
        if not hasattr(threading.current_thread(), 'supabase_client'):
            threading.current_thread().supabase_client = client_to_use
        else:
            threading.current_thread().supabase_client = client_to_use
        
        # Fetch user profile data if user_id provided
        user_profile = None
        if user_id and client_to_use:
            user_profile = fetch_user_profile(user_id, client_to_use)
        
        # Log authentication status
        if current_user:
            print(f"Chat request from authenticated user: {current_user.get('email', 'unknown')} (ID: {current_user.get('id', 'unknown')})")
        else:
            print("Chat request from unauthenticated user")
        
        # Lightweight intent detection to nudge the LLM to use tools and include IDs
        ui_lower = (user_input or "").lower()
        discovery_triggers = [
            "recommend", "suggest", "find", "show", "options", "restaurant", "places",
            "near", "around", "best", "top", "where to",
            "available", "availability", "slot", "slots", "time", "times", "book", "reserve", "reservation",
            "today", "tonight", "tomorrow", "opening", "openings"
        ]
        should_nudge = any(t in ui_lower for t in discovery_triggers)
        # Avoid nudging for cuisine list-only queries
        if ("cuisine" in ui_lower or "cuisines" in ui_lower) and any(m in ui_lower for m in ["what", "list", "available", "types", "which", "do you have"]):
            if not any(k in ui_lower for k in ["find", "recommend", "suggest", "near", "around", "restaurant", "restaurants"]):
                should_nudge = False

        # Create profile context message if user profile is available
        profile_message = None
        if user_profile:
            profile_info = []
            if user_profile.get('full_name'):
                profile_info.append(f"Name: {user_profile['full_name']}")
            if user_profile.get('allergies') and user_profile['allergies']:
                profile_info.append(f"Allergies: {', '.join(user_profile['allergies'])}")
            if user_profile.get('favorite_cuisines') and user_profile['favorite_cuisines']:
                profile_info.append(f"Favorite cuisines: {', '.join(user_profile['favorite_cuisines'])}")
            if user_profile.get('dietary_restrictions') and user_profile['dietary_restrictions']:
                profile_info.append(f"Dietary restrictions: {', '.join(user_profile['dietary_restrictions'])}")
            if user_profile.get('preferred_party_size'):
                profile_info.append(f"Preferred party size: {user_profile['preferred_party_size']}")
            if user_profile.get('loyalty_points'):
                profile_info.append(f"Loyalty points: {user_profile['loyalty_points']}")
            
            if profile_info:
                profile_message = SystemMessage(content=f"USER PROFILE: {' | '.join(profile_info)}")

        # Fetch cuisine types once per request and provide as context (no tool call)
        cuisines_message = None
        try:
            cuisines_raw = getAllCuisineTypes()
            cuisines_list = json.loads(cuisines_raw) if cuisines_raw else []
            if isinstance(cuisines_list, list) and cuisines_list:
                # Keep stable order for readability
                try:
                    cuisines_list = sorted(set(map(str, cuisines_list)), key=lambda x: x.lower())
                except Exception:
                    cuisines_list = list(dict.fromkeys(map(str, cuisines_list)))
                cuisines_message = SystemMessage(content="AVAILABLE CUISINE TYPES: " + ", ".join(cuisines_list))
        except Exception:
            cuisines_message = None

        # Create guiding message for tool usage
        guiding_message = None
        if should_nudge or user_profile:  # Create guidance if nudging needed OR user profile available
            if user_profile:
                guiding_message = SystemMessage(content=(
                    "IMPORTANT: User profile data has been provided above. Use this information for personalized recommendations.\n"
                    "For restaurant discovery: 1) Consider user's allergies, dietary restrictions, and favorite cuisines, 2) Call appropriate search tools, 3) Include up to 5 real IDs in 'RESTAURANTS_TO_SHOW:' format.\n"
                    "For availability queries: 1) Use user's preferred party size from profile, 2) Use convertRelativeDate for relative dates, 3) Find restaurant via getRestaurantsByName, 4) Use availability tools."
                ))
            else:
                guiding_message = SystemMessage(content=(
                    "RESTAURANT SEARCH RESPONSE FORMAT:\n"
                    "After using search tools, your response MUST include:\n"
                    "RESTAURANTS_TO_SHOW: id1,id2,id3,id4,id5\n\n"
                    "Example response:\n"
                    "I found amazing sushi restaurants!\n"
                    "RESTAURANTS_TO_SHOW: abc-123,def-456,ghi-789\n\n"
                    "Extract IDs from the JSON tool results (look for 'id' field in each restaurant object)."
                ))

        # Create user message
        user_message = HumanMessage(content=user_input)
        
        # Build message list based on whether we have conversation memory
        messages_to_add = []
        if profile_message:
            messages_to_add.append(profile_message)
        if cuisines_message:
            messages_to_add.append(cuisines_message)
        if guiding_message:
            messages_to_add.append(guiding_message)
        messages_to_add.append(user_message)
        
        if memory:
            # Use conversation history
            history_messages = memory.get_messages()
            current_input = {"messages": history_messages + messages_to_add}
        else:
            # Stateless mode - just the current message with context
            current_input = {"messages": messages_to_add}
        
        # Run the agent with the message(s)
        result = app.invoke(current_input)

        # Extract messages
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        
        # Save conversation to memory if provided
        if memory:
            memory.add_message(user_message)
            # Add the AI's final response to memory
            if ai_messages:
                memory.add_message(ai_messages[-1])

        # If we have a proper AI response, try to ensure IDs are present when intent suggests discovery
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if last_ai_message.content and last_ai_message.content.strip():
                text_content = last_ai_message.content.strip()
                
                # Check if response already has IDs
                if "RESTAURANTS_TO_SHOW:" not in text_content:
                    # Try to extract IDs from tool results first (more accurate than generic fallback)
                    try:
                        tool_restaurant_ids = []
                        for tool_msg in tool_messages:
                            tool_name = getattr(tool_msg, 'name', None)
                            # Check if this was a restaurant search tool
                            if tool_name in ['searchRestaurantsByFeatures', 'searchRestaurantsByMenuItem', 
                                           'searchRestaurantsByMenuCategory', 'getRestaurantsByCuisineType',
                                           'getRestaurantsByName', 'getAllRestaurants']:
                                try:
                                    tool_data = json.loads(tool_msg.content)
                                    if isinstance(tool_data, list) and tool_data:
                                        for item in tool_data:
                                            if isinstance(item, dict) and item.get('id'):
                                                tool_restaurant_ids.append(str(item['id']))
                                except Exception:
                                    pass
                        
                        # If we found restaurant IDs from tools, use them
                        if tool_restaurant_ids:
                            unique_ids = []
                            seen = set()
                            for rid in tool_restaurant_ids:
                                if rid not in seen:
                                    seen.add(rid)
                                    unique_ids.append(rid)
                            if unique_ids:
                                return text_content + "\nRESTAURANTS_TO_SHOW: " + ",".join(unique_ids[:5])
                    except Exception as e:
                        print(f"Error extracting IDs from tool results: {e}")
                    
                    # Fallback: Better intent detection for when to append generic IDs
                    intent_text = (user_input or "").lower()
                    
                    # Availability-related queries (no restaurants needed)
                    availability_markers = [
                        "available", "availability", "slot", "slots", "time", "times", "book", "reserve", "reservation",
                        "today", "tonight", "tomorrow", "opening", "openings"
                    ]
                    
                    # Actual discovery/recommendation intents
                    discovery_markers = [
                        "recommend", "suggest", "find", "show", "options", "restaurant", "places", "cuisine",
                        "near", "around", "best", "top", "where to", "looking for", "want to eat", "dinner",
                        "lunch", "breakfast", "food", "italian", "chinese", "mexican", "indian", "japanese",
                        "shisha", "outdoor", "serve", "has", "have", "offer", "offers"
                    ]
                    
                    # Greeting/general queries (no restaurants needed)
                    greeting_markers = [
                        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
                        "how are you", "what can you do", "help", "thanks", "thank you", "bye", "goodbye"
                    ]
                    
                    is_availability_query = any(t in intent_text for t in availability_markers)
                    is_discovery_query = any(t in intent_text for t in discovery_markers)
                    is_greeting_query = any(t in intent_text for t in greeting_markers)
                    
                    # Only append restaurant IDs for actual discovery queries
                    if is_discovery_query and not is_availability_query and not is_greeting_query:
                        try:
                            if supabase:
                                result = (
                                    supabase
                                    .table("restaurants")
                                    .select(restaurants_table_columns)
                                    .eq("ai_featured", True)
                                    .order("average_rating", desc=True)
                                    .limit(5)
                                    .execute()
                                )
                                items = result.data or []
                                if not items:
                                    result = (
                                        supabase
                                        .table("restaurants")
                                        .select(restaurants_table_columns)
                                        .order("ai_featured", desc=True)
                                        .order("average_rating", desc=True)
                                        .limit(5)
                                        .execute()
                                    )
                                    items = result.data or []
                                ids = [str(x.get('id')) for x in items if isinstance(x, dict) and x.get('id')]
                                ids = [i for i in ids if i][:5]
                                if ids:
                                    return text_content + "\nRESTAURANTS_TO_SHOW: " + ",".join(ids)
                        except Exception:
                            pass
                return text_content

        # Build a helpful fallback from tool outputs if available
        try:
            # Collect tool outputs by name when possible
            tool_results_by_name = {}
            for tool_msg in tool_messages:
                tool_name = getattr(tool_msg, 'name', None)
                if tool_name:
                    tool_results_by_name[tool_name] = tool_msg.content

            import json as _json

            # Prefer any restaurant list; support multiple tool outputs
            def _extract_ids(items_json: str) -> list[str]:
                try:
                    items = _json.loads(items_json)
                    if isinstance(items, list) and items:
                        ids = [str(x.get('id')) for x in items if isinstance(x, dict) and x.get('id')]
                        return [i for i in ids if i]
                except Exception:
                    return []
                return []

            all_ids: list[str] = []
            for content in tool_results_by_name.values():
                all_ids.extend(_extract_ids(content))
            # Deduplicate and limit
            seen = set()
            unique_ids = []
            for _id in all_ids:
                if _id not in seen:
                    seen.add(_id)
                    unique_ids.append(_id)
            if unique_ids:
                return "Here are some restaurants you might like.\nRESTAURANTS_TO_SHOW: " + ",".join(unique_ids[:5])

            # Cuisines list fallback
            cuisines = tool_results_by_name.get('getAllCuisineTypes')
            if cuisines:
                try:
                    cu = _json.loads(cuisines)
                    if isinstance(cu, list) and cu:
                        return "Available cuisine types: " + ", ".join(map(str, cu[:10]))
                except Exception:
                    pass
        except Exception:
            pass

        print("No AI messages found in result and no usable tool fallback")
        return "I apologize, I couldn't generate a proper response. Please try again."
            
    except Exception as e:
        print(f"Error running agent: {e}")
        # Re-raise the exception so Flask can log it properly
        raise

# Interactive chat function for testing (kept for local development)
def start_interactive_chat():
    """Start an interactive chat session with conversation memory."""
    print("🍽️ Welcome to DineMate - Your Restaurant Assistant!")
    print("Commands:")
    print("  - Type 'quit' to exit")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'history' to see conversation context")
    print("  - Type your restaurant questions naturally")
    print("Note: Conversation memory is active - I'll remember our chat!")
    print("-" * 60)
    
    # Use the global conversation memory
    global conversation_memory
    conversation_memory.clear()  # Start fresh
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using DineMate! Goodbye! 👋")
            break
        elif user_input.lower() == 'clear':
            conversation_memory.clear()
            print("🧹 Conversation history cleared!")
            continue
        elif user_input.lower() == 'history':
            messages = conversation_memory.get_messages()
            if not messages:
                print("📝 No conversation history yet.")
            else:
                print(f"📝 Conversation history ({len(messages)} messages):")
                for i, msg in enumerate(messages, 1):
                    msg_type = "You" if isinstance(msg, HumanMessage) else "DineMate"
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    print(f"  {i}. {msg_type}: {content}")
            continue
        elif not user_input:
            print("Please enter a message.")
            continue
        
        print("DineMate: ", end="", flush=True)
        response = chat_with_bot(user_input, memory=conversation_memory)
        print(response)

# Example usage for testing
if __name__ == "__main__":
    start_interactive_chat()

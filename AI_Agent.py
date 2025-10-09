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
from availability_tools import (
    check_any_time_slots as av_check_any_time_slots,
    get_available_time_slots as av_get_available_time_slots,
    get_table_options_for_slot as av_get_table_options_for_slot,
    search_time_range as av_search_time_range,
)
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
You are a specialized restaurant assistant for Plate your name is DineMate, a restaurant reservation app.

## YOUR ROLE
Your ONLY responsibilities are to:
1. Help users find restaurants based on their preferences
2. Provide personalized recommendations using user profile data (allergies, dietary restrictions, favorite cuisines, preferred party size)
3. Provide information about restaurants (cuisine, price, features, ratings)
4. Answer questions about restaurant availability and booking policies
5. Maintain a friendly and professional tone

## AVAILABLE DATA
You have access to comprehensive restaurant information:
- Restaurant names, descriptions, and cuisine types
- Price ranges (1-4) and average ratings
- Features: outdoor seating, parking, shisha availability
- Opening hours and booking policies
- Real-time availability data

## RESPONSE FORMAT
When showing restaurants to users, ALWAYS use this exact format:

**Structure:**
1. Provide your conversational response
2. Add "RESTAURANTS_TO_SHOW:" on a new line
3. List restaurant IDs separated by commas (max 5 IDs)

**Example:**
```
I found some great Italian restaurants for you!
RESTAURANTS_TO_SHOW: restaurant-1,restaurant-2,restaurant-3
```

## RESTAURANT RECOMMENDATION RULES
- **ALWAYS** use database tools first - never guess or invent restaurant IDs
- **ALWAYS** prioritize restaurants where ai_featured = true, then by highest average_rating
- **LIMIT** to maximum 5 restaurant IDs
- **CALL** finishedUsingTools after completing any tool usage

## WORKFLOW GUIDELINES

### For Restaurant Discovery/Recommendations:
1. Use appropriate search tool (by cuisine, name, featured, or advanced filters)
2. **IF USER PROFILE PROVIDED:** Consider user's allergies, dietary restrictions, and favorite cuisines
3. Filter recommendations based on user's allergies and dietary restrictions when available
4. Prioritize user's favorite cuisines when provided
5. Format response with RESTAURANTS_TO_SHOW
6. Call finishedUsingTools

### For User Profile-Based Personalization:
- User profile data (if available) will be provided in conversation context
- Always consider allergies and dietary_restrictions when recommending restaurants
- Use preferred_party_size for availability queries if not specified by user
- Mention user's favorite_cuisines in recommendations when relevant
- Reference loyalty_points for special offers or tier-based suggestions

### For Availability Questions:
1. **FIRST:** Convert relative dates using convertRelativeDate tool
   - "today", "tomorrow", "tonight" → YYYY-MM-DD format
2. **SECOND:** Find restaurant using getRestaurantsByName
3. **THIRD:** Use availability tools with converted date:
   - checkAnyTimeSlots (yes/no availability)
   - getAvailableTimeSlots (list specific times)
   - getTableOptionsForSlot (table details for specific time)
   - searchTimeRange (explore time windows)
4. **PARTY SIZE:** Use user's preferred_party_size from profile if available, otherwise assume 2 people (state this clearly)
5. **FINISH:** Call finishedUsingTools

## STRICT CONSTRAINTS
**SCOPE LIMITATIONS:**
- ONLY answer restaurant, dining, and reservation questions
- DO NOT provide code, programming solutions, or technical implementations
- DO NOT answer questions outside restaurant assistance scope
- Politely redirect non-restaurant topics to restaurant-related subjects

**DATA INTEGRITY:**
- Base ALL responses on available restaurant data
- Use ONLY the tools provided for restaurant data lookup
- NEVER fabricate restaurant IDs or information

**RESPONSE POLICY:**
- For tool-based responses: ALWAYS call finishedUsingTools when complete
- For direct responses (general service questions): respond without tools
- Keep all responses focused on restaurant discovery and booking assistance
"""
restaurants_table_columns:str = "id, name, description, address, tags, opening_time, closing_time, cuisine_type, price_range, average_rating, dietary_options, ambiance_tags, outdoor_seating, ai_featured"
@tool
def finishedUsingTools() -> str:
    """Call this when you're done using tools and ready to respond."""
    print("AI finished using tools")
    return "Tools usage completed. Ready to provide response."

tools.append(finishedUsingTools)

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

@tool
def getAllCuisineTypes() -> str:
    """Return the unique cuisine types available in the application"""
    print("AI is looking for cuisine types")
    try:
        if not supabase:
            return json.dumps([])
        result = supabase.table("restaurants").select("cuisine_type").execute()
        cuisineTypes = result.data
        
        if not cuisineTypes:
            return "Currently we have no cuisine types available"
        
        # Extract unique cuisine types
        unique_cuisines = list(set([item['cuisine_type'] for item in cuisineTypes if item.get('cuisine_type')]))
        print(f"Found cuisine types: {unique_cuisines}")
        return json.dumps(unique_cuisines)
    except Exception as e:
        print(f"Error fetching cuisine types: {e}")
        return "Error retrieving cuisine types"

tools.append(getAllCuisineTypes)

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
    """Request all restaurants with all their info from the database"""
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
            .limit(50)
            .execute()
        )
        restaurants = result.data

        if not restaurants:
            return "No restaurants found"
        print("the restaurants found are: "+str(restaurants))
        return json.dumps(restaurants)
    
    except Exception as e:
        print(f"Error fetching restaurants: {e}")
        return "Error retrieving restaurants"

tools.append(getAllRestaurants)

@tool
def getFeaturedRestaurants(limit: int = 10) -> str:
    """Return featured restaurants prioritized by rating. Limit defaults to 10."""
    print("AI is looking for featured restaurants")
    try:
        if not supabase:
            return json.dumps([])
        lim = max(1, min(int(limit or 10), 100))
        result = (
            supabase
            .table("restaurants")
            .select(restaurants_table_columns)
            .eq("ai_featured", True)
            .order("average_rating", desc=True)
            .limit(lim)
            .execute()
        )
        restaurants = result.data
        if not restaurants:
            return json.dumps([])
        return json.dumps(restaurants)
    except Exception as e:
        print(f"Error fetching featured restaurants: {e}")
        return json.dumps([])

tools.append(getFeaturedRestaurants)

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
def searchRestaurantsAdvanced(filters_json: str) -> str:
    """Advanced restaurant search. Accepts a JSON string with optional fields: 
    {"cuisine":"italian","price_min":1,"price_max":3,"rating_min":4,"has_outdoor":true,"tags":["shisha","parking"],"ambiance":["romantic"]}
    Returns a JSON list of restaurants sorted by featured then rating.
    """
    print(f"AI is running advanced restaurant search with filters: {filters_json}")
    try:
        if not supabase:
            return json.dumps([])
        parsed = {}
        try:
            parsed = json.loads(filters_json) if filters_json else {}
        except Exception:
            parsed = {}

        query = supabase.table("restaurants").select(restaurants_table_columns)

        cuisine = (parsed.get("cuisine") or "").strip()
        if cuisine:
            query = query.ilike("cuisine_type", f"%{cuisine}%")

        price_min = parsed.get("price_min")
        price_max = parsed.get("price_max")
        if isinstance(price_min, (int, float)):
            query = query.gte("price_range", int(price_min))
        if isinstance(price_max, (int, float)):
            query = query.lte("price_range", int(price_max))

        rating_min = parsed.get("rating_min")
        if isinstance(rating_min, (int, float)):
            query = query.gte("average_rating", float(rating_min))

        has_outdoor = parsed.get("has_outdoor")
        if isinstance(has_outdoor, bool):
            query = query.eq("outdoor_seating", has_outdoor)

        tags = parsed.get("tags")
        if isinstance(tags, list) and tags:
            try:
                query = query.contains("tags", tags)
            except Exception:
                # Fallback to text match if contains not available
                for t in tags:
                    query = query.ilike("tags", f"%{t}%")

        ambiance = parsed.get("ambiance")
        if isinstance(ambiance, list) and ambiance:
            try:
                query = query.contains("ambiance_tags", ambiance)
            except Exception:
                for a in ambiance:
                    query = query.ilike("ambiance_tags", f"%{a}%")

        limit = parsed.get("limit")
        lim = max(1, min(int(limit or 50), 100))

        result = (
            query
            .order("ai_featured", desc=True)
            .order("average_rating", desc=True)
            .limit(lim)
            .execute()
        )
        items = result.data or []
        return json.dumps(items)
    except Exception as e:
        print(f"Error in advanced search: {e}")
        return json.dumps([])

tools.append(searchRestaurantsAdvanced)

# -----------------------------
# Availability tools (backend service key based)
# -----------------------------

@tool
def checkAnyTimeSlots(restaurant_id: str, date: str, party_size: int, user_id: Optional[str] = None) -> str:
    """Return {"available": bool} if at least one slot exists for the given date and party size."""
    try:
        available = av_check_any_time_slots(restaurant_id, date, int(party_size), user_id)
        return json.dumps({"available": bool(available)})
    except Exception as e:
        return json.dumps({"error": str(e)})

tools.append(checkAnyTimeSlots)

@tool
def getAvailableTimeSlots(restaurant_id: str, date: str, party_size: int, user_id: Optional[str] = None) -> str:
    """Return a JSON list of {time: 'HH:MM', available: true} slots for the day."""
    try:
        slots = av_get_available_time_slots(restaurant_id, date, int(party_size), user_id)
        return json.dumps(slots)
    except Exception as e:
        return json.dumps({"error": str(e)})

tools.append(getAvailableTimeSlots)

@tool
def getTableOptionsForSlot(restaurant_id: str, date: str, time: str, party_size: int, user_id: Optional[str] = None) -> str:
    """Return table options for a specific time slot, or null if none."""
    try:
        # Note: The underlying function may not use user_id, but we keep it for consistency
        options = av_get_table_options_for_slot(restaurant_id, date, time, int(party_size))
        return json.dumps(options)
    except Exception as e:
        return json.dumps({"error": str(e)})

tools.append(getTableOptionsForSlot)

@tool
def searchTimeRange(restaurant_id: str, date: str, start_time: str, end_time: str, party_size: int, user_id: Optional[str] = None) -> str:
    """Return best table options for all available slots within a time range on a given date."""
    try:
        results = av_search_time_range(restaurant_id, date, start_time, end_time, int(party_size), user_id)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

tools.append(searchTimeRange)

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
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
    """Determine whether to continue with tools or end the conversation"""
    last_message = state["messages"][-1]

    
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', []) or []
        print(f"Tool calls found: {len(tool_calls)}")
        
        # If there are tool calls, check if finishedUsingTools was called
        for call in tool_calls:
            print(f"Tool call: {call}")
            if call["name"] == "finishedUsingTools":
                print("✅ AI called finishedUsingTools tool - ending")
                return "end"
        
        # If there are other tool calls, continue to tools
        if tool_calls:
            print("🔧 AI has tool calls - continuing to tools")
            return "continue"
        
        # If no tool calls and has content, end
        if last_message.content:
            print("💬 AI has content but no tool calls - ending")
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
            "recommend", "suggest", "find", "show", "options", "restaurant", "places", "cuisine",
            "near", "around", "best", "top", "where to",
            "available", "availability", "slot", "slots", "time", "times", "book", "reserve", "reservation",
            "today", "tonight", "tomorrow", "opening", "openings"
        ]
        should_nudge = any(t in ui_lower for t in discovery_triggers)

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

        # Create guiding message for tool usage
        guiding_message = None
        if should_nudge or user_profile:  # Create guidance if nudging needed OR user profile available
            if user_profile:
                guiding_message = SystemMessage(content=(
                    "IMPORTANT: User profile data has been provided above. Use this information for personalized recommendations.\n"
                    "For restaurant discovery: 1) Consider user's allergies, dietary restrictions, and favorite cuisines, 2) Call appropriate search tools, 3) Include up to 5 real IDs in 'RESTAURANTS_TO_SHOW:' format.\n"
                    "For availability queries: 1) Use user's preferred party size from profile, 2) Use convertRelativeDate for relative dates, 3) Find restaurant via getRestaurantsByName, 4) Use availability tools.\n"
                    "Always call finishedUsingTools when done."
                ))
            else:
                guiding_message = SystemMessage(content=(
                    "For this request, if it's about discovering restaurants: call the appropriate search tools and include up to 5 real IDs in a line starting with 'RESTAURANTS_TO_SHOW:'. Prioritize featured and highly-rated restaurants.\n"
                    "If it's about availability for a specific restaurant: 1) FIRST use convertRelativeDate for any relative dates (today, tomorrow, etc.), 2) locate the restaurant via getRestaurantsByName, 3) use availability tools with the converted date. Assume party size 2 if unspecified; state assumptions. When done with tools, call finishedUsingTools."
                ))

        # Create user message
        user_message = HumanMessage(content=user_input)
        
        # Build message list based on whether we have conversation memory
        messages_to_add = []
        if profile_message:
            messages_to_add.append(profile_message)
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
                if "RESTAURANTS_TO_SHOW:" not in text_content:
                    # Better intent detection: only append restaurant IDs for actual discovery requests
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
                        "lunch", "breakfast", "food", "italian", "chinese", "mexican", "indian", "japanese"
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
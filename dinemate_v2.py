"""
DineMate v2 - Structured Multi-Stage LangGraph Restaurant Assistant

This is a reimplementation of the restaurant assistant with a more explicit pipeline:
1. classifier_node: Detects user intent (restaurant_search, smalltalk, info_question)
2. slot_filler_node: Extracts structured search filters from user message
3. tool_runner_node: Executes tools deterministically based on filters (no LLM)
4. response_node: Generates natural language response with personality
5. smalltalk_responder_node: Handles casual conversation

Key differences from v1:
- No llm.bind_tools - tools are called explicitly in Python
- State tracks intent, search_type, filters, and results explicitly
- Separation of concerns: classification → extraction → execution → generation
"""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from supabase import Client
from datetime import datetime
from dateutil import tz
import json
import threading

# Import utilities from the original agent file
from AI_Agent import (
    get_supabase_client,
    ConversationMemory,
    create_conversation_memory,
    fetch_user_profile,
    getAllCuisineTypes,
    getAllMenuCategories,
    # Import the tool functions directly (not as @tool decorated)
    convertRelativeDate,
    getRestaurantsByCuisineType,
    getAllRestaurants,
    getRestaurantsByName,
    searchRestaurantsByFeatures,
    searchRestaurantsByMenuItem,
    searchRestaurantsByMenuCategory,
)

load_dotenv()

# Get timezone for consistent date handling
_LOCAL_TZ = tz.gettz(os.getenv("AVAILABILITY_TZ", "Asia/Beirut")) or tz.UTC

# ========================================
# State Definitions
# ========================================

class SearchFilters(TypedDict, total=False):
    """Structured search filters extracted from user query"""
    cuisine: Optional[str]
    dish: Optional[str]
    menu_category: Optional[str]
    restaurant_name: Optional[str]
    outdoor_seating: Optional[bool]
    shisha_available: Optional[bool]
    min_rating: Optional[float]
    price_range: Optional[int]
    relative_date: Optional[str]
    party_size: Optional[int]


class AgentState(TypedDict):
    """V2 Agent State with explicit tracking of intent and search parameters"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Intent classification
    intent: str                      # "restaurant_search", "smalltalk", "info_question"
    search_type: str                 # "cuisine", "dish", "features", "name", "menu_category", "none"
    wants_restaurants: bool          # Whether user wants restaurant recommendations
    is_refinement: bool              # Whether user is refining previous search
    
    # Category selection (Cerebras-powered)
    selected_categories: List[str]   # Relevant menu categories filtered by Cerebras
    
    # Search filters
    filters: Dict[str, Any]          # Extracted search filters
    previous_filters: Dict[str, Any] # Previous search filters for refinement
    
    # Results
    restaurants: List[dict]          # Tool execution results
    previous_restaurants: List[dict] # Previous search results for refinement
    
    # User context
    user_profile: Optional[dict]     # User preferences, allergies, etc.


# ========================================
# LLM Initialization
# ========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Cerebras API configuration (OpenAI-compatible endpoint)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_AVAILABLE = bool(CEREBRAS_API_KEY)

if CEREBRAS_AVAILABLE:
    print("✅ Cerebras API configured for category selection")
else:
    print("⚠️ Cerebras API key not found, will use Gemini for all tasks")


def call_cerebras_api(messages: List[dict], max_tokens: int = 500) -> str:
    """Call Cerebras API directly using requests (no langchain dependency).
    
    Args:
        messages: List of {"role": "system"|"user", "content": "..."}
        max_tokens: Maximum tokens to generate
        
    Returns:
        Response content as string
    """
    import requests
    
    if not CEREBRAS_AVAILABLE:
        raise Exception("Cerebras API key not configured")
    
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3.1-8b",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    
    response = requests.post(CEREBRAS_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


# ========================================
# Helper Functions
# ========================================

def append_restaurants_to_show(text: str, restaurants: List[dict]) -> str:
    """Append RESTAURANTS_TO_SHOW line with restaurant IDs to response text.
    
    This is a critical requirement for the frontend to display restaurants.
    """
    ids = [str(r.get("id")) for r in restaurants if r.get("id")]
    unique_ids = []
    seen = set()
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            unique_ids.append(rid)
    
    if unique_ids:
        return text.strip() + "\nRESTAURANTS_TO_SHOW: " + ",".join(unique_ids[:5])
    return text.strip()


def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    content = content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Content: {content}")
        return {}


# ========================================
# Node 1: Classifier
# ========================================

CLASSIFIER_SYSTEM_PROMPT = """You are a classification assistant for a restaurant discovery app.

Analyze the user's message and classify it into:

1. **intent**: One of:
   - "restaurant_search": User wants restaurant recommendations/discovery
   - "smalltalk": Casual conversation (greetings, how are you, etc.)
   - "info_question": Questions about cuisine types, features, or general info

2. **search_type**: One of:
   - "cuisine": User mentions a cuisine type (Italian, Chinese, Lebanese, etc.)
   - "dish": User mentions a specific dish or menu item (sushi, pasta, tacos, etc.)
   - "menu_category": User mentions a menu category (Desserts, Appetizers, Cocktails, etc.)
   - "name": User mentions a specific restaurant name
   - "features": User mentions features (outdoor seating, shisha, price range, rating)
   - "none": No specific search criteria

3. **wants_restaurants**: Boolean
   - true if user wants restaurant recommendations
   - false for info questions or smalltalk

4. **is_refinement**: Boolean
   - true if user is refining/filtering previous results ("with outdoor seating", "cheaper ones", "highly rated")
   - false if it's a completely new search

Respond with JSON only:
{
  "intent": "...",
  "search_type": "...",
  "wants_restaurants": true/false,
  "is_refinement": true/false
}

Examples:

User: "Find Italian restaurants"
Response: {"intent": "restaurant_search", "search_type": "cuisine", "wants_restaurants": true, "is_refinement": false}

User: "Where can I get sushi?"
Response: {"intent": "restaurant_search", "search_type": "dish", "wants_restaurants": true, "is_refinement": false}

User: "I would like it outdoor" (after previous search)
Response: {"intent": "restaurant_search", "search_type": "features", "wants_restaurants": true, "is_refinement": true}

User: "Show me cheaper ones" (after previous results)
Response: {"intent": "restaurant_search", "search_type": "features", "wants_restaurants": true, "is_refinement": true}

User: "Hi, how are you?"
Response: {"intent": "smalltalk", "search_type": "none", "wants_restaurants": false, "is_refinement": false}

User: "What cuisine types are available?"
Response: {"intent": "info_question", "search_type": "none", "wants_restaurants": false, "is_refinement": false}

User: "Show me places with outdoor seating and shisha"
Response: {"intent": "restaurant_search", "search_type": "features", "wants_restaurants": true, "is_refinement": false}

User: "Tell me about Babel restaurant"
Response: {"intent": "restaurant_search", "search_type": "name", "wants_restaurants": true, "is_refinement": false}
"""


def classifier_node(state: AgentState) -> AgentState:
    """Classify user intent and search type using LLM."""
    print("🔍 [CLASSIFIER] Analyzing user intent...")
    
    # Get the latest user message
    messages = state["messages"]
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg
            break
    
    if not latest_user_msg:
        print("⚠️ [CLASSIFIER] No user message found")
        return {
            "intent": "smalltalk",
            "search_type": "none",
            "wants_restaurants": False
        }
    
    # Call LLM for classification
    classification_messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        latest_user_msg
    ]
    
    response = llm.invoke(classification_messages)
    result = parse_json_response(response.content)
    
    intent = result.get("intent", "smalltalk")
    search_type = result.get("search_type", "none")
    wants_restaurants = result.get("wants_restaurants", False)
    is_refinement = result.get("is_refinement", False)
    
    print(f"✅ [CLASSIFIER] Intent: {intent}, Search: {search_type}, Wants restaurants: {wants_restaurants}, Refinement: {is_refinement}")
    
    return {
        "intent": intent,
        "search_type": search_type,
        "wants_restaurants": wants_restaurants,
        "is_refinement": is_refinement
    }


# ========================================
# Node 2: Category Selector (Cerebras-powered)
# ========================================

CATEGORY_SELECTOR_SYSTEM_PROMPT = """You are a fast category filtering assistant.

Given a user's restaurant query and a complete list of menu categories, select ONLY the categories that are relevant to their request.

**Rules:**
- Select categories that directly relate to the user's query
- Include partial matches (e.g., "sushi" should match "Sushi & Sashimi")
- If unsure, include the category (better to over-include than miss)
- Return a JSON array of category names only
- If no specific food/category mentioned, return empty array []

**Examples:**

User: "Show me places with good sushi"
Categories: ["Sushi & Sashimi", "Desserts", "Japanese Dishes", "Appetizers"]
Response: ["Sushi & Sashimi", "Japanese Dishes"]

User: "I want desserts"
Categories: ["DESSERTS", "Sweets", "Main Courses", "Appetizers"]
Response: ["DESSERTS", "Sweets"]

User: "Find Italian restaurants"
Categories: ["Pizza", "Pasta", "Sushi", "Desserts"]
Response: []  // No category filtering needed, it's a cuisine search

User: "Restaurants with cocktails"
Categories: ["Cocktails", "Cocktails & Mocktails", "Beverages", "Desserts"]
Response: ["Cocktails", "Cocktails & Mocktails", "Beverages"]

Respond with ONLY a JSON array, nothing else.
"""


def category_selector_node(state: AgentState) -> AgentState:
    """Select relevant menu categories using Cerebras Llama 3.1 8B (fast & free)."""
    print("🎯 [CATEGORY_SELECTOR] Filtering relevant categories with Cerebras...")
    
    messages = state["messages"]
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg
            break
    
    if not latest_user_msg:
        print("⚠️ [CATEGORY_SELECTOR] No user message found")
        return {"selected_categories": []}
    
    # Fetch ALL menu categories
    try:
        categories_raw = getAllMenuCategories()
        all_categories = json.loads(categories_raw) if categories_raw else []
        
        if not all_categories:
            print("⚠️ [CATEGORY_SELECTOR] No categories available")
            return {"selected_categories": []}
        
        print(f"📋 [CATEGORY_SELECTOR] Analyzing {len(all_categories)} categories...")
        
        # Build prompt with all categories
        prompt_text = f"{CATEGORY_SELECTOR_SYSTEM_PROMPT}\n\nAVAILABLE_CATEGORIES: {json.dumps(all_categories)}\n\nUser Query: {latest_user_msg.content}"
        
        # Use Cerebras for fast, free category selection
        if CEREBRAS_AVAILABLE:
            try:
                cerebras_messages = [
                    {"role": "system", "content": CATEGORY_SELECTOR_SYSTEM_PROMPT},
                    {"role": "system", "content": f"AVAILABLE_CATEGORIES: {json.dumps(all_categories)}"},
                    {"role": "user", "content": latest_user_msg.content}
                ]
                response_text = call_cerebras_api(cerebras_messages, max_tokens=500)
                selected_categories = parse_json_response(response_text)
            except Exception as e:
                print(f"⚠️ [CATEGORY_SELECTOR] Cerebras API error: {e}, falling back to Gemini")
                # Fallback to Gemini
                fallback_messages = [
                    SystemMessage(content=prompt_text),
                    latest_user_msg
                ]
                response = llm.invoke(fallback_messages)
                selected_categories = parse_json_response(response.content)
        else:
            # Use Gemini if Cerebras not available
            fallback_messages = [
                SystemMessage(content=prompt_text),
                latest_user_msg
            ]
            response = llm.invoke(fallback_messages)
            selected_categories = parse_json_response(response.content)
        
        # Handle both array and object responses
        if isinstance(selected_categories, list):
            result = selected_categories
        elif isinstance(selected_categories, dict) and 'categories' in selected_categories:
            result = selected_categories['categories']
        else:
            result = []
        
        print(f"✅ [CATEGORY_SELECTOR] Selected {len(result)} relevant categories: {result[:5]}")
        
        # Store in state for slot_filler to use
        return {"selected_categories": result}
        
    except Exception as e:
        print(f"❌ [CATEGORY_SELECTOR] Error: {e}")
        # On error, pass empty list (slot_filler will get all categories)
        return {"selected_categories": []}


# ========================================
# Node 3: Slot Filler
# ========================================

SLOT_FILLER_SYSTEM_PROMPT = """You are a data extraction assistant for a restaurant search system.

Extract structured search filters from the user's message.

You will be provided with AVAILABLE_MENU_CATEGORIES from the database.
When extracting menu_category, match the user's intent to the closest available category name from the list.
Use fuzzy matching - if user says "sushi", match it to categories like "Sushi & Sashimi" or "Japanese Sushi".

Output JSON with these fields (all optional):
{
  "cuisine": "Italian" | "Chinese" | "Lebanese" | etc. (capitalize first letter),
  "dish": "sushi" | "pasta" | "tacos" | etc. (specific menu item),
  "menu_category": Match to closest available category from AVAILABLE_MENU_CATEGORIES,
  "restaurant_name": "exact or partial restaurant name",
  "outdoor_seating": true | false | null,
  "shisha_available": true | false | null,
  "min_rating": 4.0 | 4.5 | etc. | null,
  "price_range": 1-4 (1=budget, 4=expensive) | null,
  "relative_date": "today" | "tomorrow" | "next week" | etc. | null,
  "party_size": 2 | 4 | etc. | null
}

Only include fields that are explicitly mentioned or clearly implied.

Examples:

User: "Find Italian restaurants with outdoor seating"
Response: {"cuisine": "Italian", "outdoor_seating": true}

User: "Where can I get good sushi for 4 people?"
Available categories: ["Sushi & Sashimi", "Japanese Dishes", "Desserts"]
Response: {"dish": "sushi", "menu_category": "Sushi & Sashimi", "party_size": 4}

User: "Show me highly rated places with shisha"
Response: {"shisha_available": true, "min_rating": 4.0}

User: "I want cheap Italian food"
Response: {"cuisine": "Italian", "price_range": 1}

User: "Restaurants for tomorrow"
Response: {"relative_date": "tomorrow"}

User: "Tell me about Babel"
Response: {"restaurant_name": "Babel"}

User: "Places with good desserts"
Available categories: ["Desserts", "Sweets & Pastries", "Sweet Treats"]
Response: {"menu_category": "Desserts"}

User: "I want cocktails"
Available categories: ["Cocktails & Mocktails", "Beverages", "Drinks"]
Response: {"menu_category": "Cocktails & Mocktails"}
"""


def slot_filler_node(state: AgentState) -> AgentState:
    """Extract structured search filters from user message using pre-selected categories."""
    print("📋 [SLOT_FILLER] Extracting search filters...")
    
    # Get the latest user message
    messages = state["messages"]
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg
            break
    
    if not latest_user_msg:
        print("⚠️ [SLOT_FILLER] No user message found")
        return {"filters": {}}
    
    # Check if this is a refinement of previous search
    is_refinement = state.get("is_refinement", False)
    previous_filters = state.get("previous_filters", {})
    
    if is_refinement and previous_filters:
        print(f"🔄 [SLOT_FILLER] Refinement mode - merging with previous filters: {previous_filters}")
    
    # Use pre-selected categories from category_selector_node
    selected_categories = state.get("selected_categories", [])
    
    menu_categories_context = None
    if selected_categories:
        # Use Cerebras-filtered categories (optimized)
        menu_categories_context = SystemMessage(
            content=f"AVAILABLE_MENU_CATEGORIES: {', '.join(selected_categories)}"
        )
        print(f"📋 [SLOT_FILLER] Using {len(selected_categories)} pre-selected categories from Cerebras")
    else:
        # Fallback: fetch all categories if Cerebras filtering didn't run or returned empty
        try:
            categories_raw = getAllMenuCategories()
            categories_list = json.loads(categories_raw) if categories_raw else []
            if isinstance(categories_list, list) and categories_list:
                menu_categories_context = SystemMessage(
                    content=f"AVAILABLE_MENU_CATEGORIES: {', '.join(categories_list)}"
                )
                print(f"📋 [SLOT_FILLER] Using all {len(categories_list)} categories (fallback)")
        except Exception as e:
            print(f"⚠️ [SLOT_FILLER] Could not load menu categories: {e}")
            menu_categories_context = None
    
    # Call LLM for slot filling
    slot_messages = [SystemMessage(content=SLOT_FILLER_SYSTEM_PROMPT)]
    if menu_categories_context:
        slot_messages.append(menu_categories_context)
    slot_messages.append(latest_user_msg)
    
    response = llm.invoke(slot_messages)
    new_filters = parse_json_response(response.content)
    
    # Merge with previous filters if this is a refinement
    if is_refinement and previous_filters:
        # Start with previous filters
        merged_filters = previous_filters.copy()
        # Override/add new filters
        merged_filters.update(new_filters)
        print(f"✅ [SLOT_FILLER] Merged filters: {merged_filters} (previous: {previous_filters}, new: {new_filters})")
        final_filters = merged_filters
    else:
        print(f"✅ [SLOT_FILLER] Extracted filters: {new_filters}")
        final_filters = new_filters
    
    # Store current filters as previous for next interaction
    return {
        "filters": final_filters,
        "previous_filters": final_filters
    }


# ========================================
# Node 3: Tool Runner (Pure Python)
# ========================================

def tool_runner_node(state: AgentState) -> AgentState:
    """Execute tools deterministically based on search_type and filters.
    
    This node does NOT use the LLM - it's pure Python logic.
    Prioritizes specific filters (menu_category, dish, cuisine) over generic search_type.
    """
    print("🔧 [TOOL_RUNNER] Executing tools...")
    
    search_type = state.get("search_type", "none")
    filters = state.get("filters", {})
    
    restaurants = []
    
    try:
        # Handle relative date conversion first if present
        if filters.get("relative_date"):
            print(f"📅 [TOOL_RUNNER] Converting relative date: {filters['relative_date']}")
            converted_date = convertRelativeDate.invoke({"relative_date": filters["relative_date"]})
            filters["converted_date"] = converted_date
            print(f"✅ [TOOL_RUNNER] Converted to: {converted_date}")
        
        # Route to appropriate tool based on filters (prioritize specific filters over search_type)
        # Priority: menu_category > dish > cuisine > name > features > all
        
        if filters.get("menu_category"):
            # Menu category has highest priority (handles "places with sushi", "restaurants with desserts")
            print(f"📖 [TOOL_RUNNER] Searching by menu category: {filters['menu_category']}")
            result_json = searchRestaurantsByMenuCategory.invoke({"category_name": filters["menu_category"]})
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
            
        elif filters.get("dish"):
            print(f"🍕 [TOOL_RUNNER] Searching by dish: {filters['dish']}")
            result_json = searchRestaurantsByMenuItem.invoke({"query": filters["dish"]})
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
            
        elif filters.get("cuisine"):
            print(f"🍽️ [TOOL_RUNNER] Searching by cuisine: {filters['cuisine']}")
            result_json = getRestaurantsByCuisineType.invoke({"cuisineType": filters["cuisine"]})
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
            
        elif filters.get("restaurant_name"):
            print(f"🏪 [TOOL_RUNNER] Searching by name: {filters['restaurant_name']}")
            result_json = getRestaurantsByName.invoke({"query": filters["restaurant_name"]})
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
            
        elif search_type == "features" or any([
            filters.get("outdoor_seating"),
            filters.get("shisha_available"),
            filters.get("min_rating"),
            filters.get("price_range")
        ]):
            print(f"🎯 [TOOL_RUNNER] Searching by features: {filters}")
            result_json = searchRestaurantsByFeatures.invoke({
                "outdoor_seating": filters.get("outdoor_seating"),
                "shisha_available": filters.get("shisha_available"),
                "min_rating": filters.get("min_rating"),
                "price_range": filters.get("price_range")
            })
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
            
        else:
            # Default: get all restaurants
            print("🌐 [TOOL_RUNNER] Getting all restaurants (default)")
            result_json = getAllRestaurants.invoke({})
            restaurants = json.loads(result_json) if isinstance(result_json, str) else result_json
        
        # Ensure restaurants is a list
        if not isinstance(restaurants, list):
            restaurants = []
        
        print(f"✅ [TOOL_RUNNER] Found {len(restaurants)} restaurants")
        
    except Exception as e:
        print(f"❌ [TOOL_RUNNER] Error executing tools: {e}")
        restaurants = []
    
    # Store results for potential refinement in next turn
    return {
        "restaurants": restaurants,
        "previous_restaurants": restaurants
    }


# ========================================
# Node 4: Response Generator
# ========================================

RESPONSE_SYSTEM_PROMPT = """You are DineMate, a warm and enthusiastic restaurant discovery assistant.

**YOUR PERSONALITY:**
- Conversational & natural (like a friendly foodie friend, not a robot)
- Enthusiastic about food and dining
- Concise but warm (3-5 sentences max)
- Use emojis naturally but don't overdo it

**YOUR TASK:**
You will be given:
1. The user's question/request
2. A JSON list of restaurant results (if any)
3. User profile data (allergies, favorites, etc.) if available

Generate a friendly, natural response that:
- Shows enthusiasm about helping
- Describes 2-3 top restaurants with personality (what makes them special, not just features)
- Weaves in user preferences naturally if available
- Invites them to explore more

**IMPORTANT:**
- Do NOT include "RESTAURANTS_TO_SHOW:" in your response - that's added automatically
- Keep it conversational, not a data dump
- Highlight what makes places special, not just list features

**Examples:**

User: "Italian restaurants"
Restaurants: [3 Italian places with details]
Response: "Ooh Italian! 🍝 I found some incredible spots for you! La Dolce Vita has the most authentic Roman-style pasta, Basilico is perfect for modern Italian with a twist, and Trattoria Bella is cozy family-run heaven. Which vibe are you after?"

User: "Where can I get sushi?"
Restaurants: [2 sushi places]
Response: "Sushi lover! 🍣 I've got two amazing places - Sakura does traditional omakase that's absolutely divine, and Fusion Sake has creative rolls with Lebanese twists. Both get rave reviews!"

User: "Show me places with outdoor seating"
Restaurants: [4 places with outdoor seating]
User Profile: allergies=["peanuts"]
Response: "Love outdoor dining! 🌿 I found some gorgeous spots - and I made sure they're all peanut-free for you. Garden Terrace has sunset views, Rooftop Lounge is perfect for drinks and small plates, and Green House does farm-to-table in a garden setting. Want to check them out?"
"""


def response_node(state: AgentState) -> AgentState:
    """Generate natural language response with DineMate personality."""
    print("💬 [RESPONSE] Generating friendly response...")
    
    messages = state["messages"]
    restaurants = state.get("restaurants", [])
    user_profile = state.get("user_profile")
    
    # Get latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg
            break
    
    if not latest_user_msg:
        print("⚠️ [RESPONSE] No user message found")
        return {"messages": [AIMessage(content="I'm here to help you find great restaurants! What are you craving?")]}
    
    # Build context messages
    context_messages = [SystemMessage(content=RESPONSE_SYSTEM_PROMPT)]
    
    # Add restaurant results as context
    if restaurants:
        restaurants_json = json.dumps(restaurants[:5], indent=2)
        context_messages.append(SystemMessage(content=f"RESTAURANT_RESULTS:\n{restaurants_json}"))
    else:
        context_messages.append(SystemMessage(content="RESTAURANT_RESULTS: No restaurants found matching the criteria."))
    
    # Add user profile as context
    if user_profile:
        profile_info = []
        if user_profile.get('full_name'):
            profile_info.append(f"Name: {user_profile['full_name']}")
        if user_profile.get('allergies'):
            profile_info.append(f"Allergies: {', '.join(user_profile['allergies'])}")
        if user_profile.get('favorite_cuisines'):
            profile_info.append(f"Favorite cuisines: {', '.join(user_profile['favorite_cuisines'])}")
        if user_profile.get('dietary_restrictions'):
            profile_info.append(f"Dietary restrictions: {', '.join(user_profile['dietary_restrictions'])}")
        
        if profile_info:
            context_messages.append(SystemMessage(content=f"USER_PROFILE: {' | '.join(profile_info)}"))
    
    # Add user message
    context_messages.append(latest_user_msg)
    
    # Generate response
    response = llm.invoke(context_messages)
    response_text = response.content.strip()
    
    # Append RESTAURANTS_TO_SHOW line if we have restaurants
    if restaurants:
        response_text = append_restaurants_to_show(response_text, restaurants)
    
    print(f"✅ [RESPONSE] Generated response ({len(response_text)} chars)")
    
    return {"messages": [AIMessage(content=response_text)]}


# ========================================
# Node 5: Smalltalk Responder
# ========================================

SMALLTALK_SYSTEM_PROMPT = """You are DineMate, a friendly restaurant discovery assistant.

The user is just making small talk or casual conversation.

Respond briefly and warmly (1-2 sentences), and gently invite them to ask about food or restaurants.

Examples:

User: "Hi!"
Response: "Hey there! 👋 I'm DineMate, your friendly restaurant guide. What kind of food experience are you looking for today?"

User: "How are you?"
Response: "I'm great, thanks for asking! 😊 Always excited to help people find amazing food. What are you craving?"

User: "Thank you!"
Response: "You're so welcome! Happy to help anytime. Let me know if you need more restaurant suggestions! 🍽️"

Keep it natural, warm, and brief!
"""


def smalltalk_responder_node(state: AgentState) -> AgentState:
    """Handle casual conversation without restaurant search."""
    print("💭 [SMALLTALK] Handling casual conversation...")
    
    messages = state["messages"]
    
    # Get latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg
            break
    
    if not latest_user_msg:
        return {"messages": [AIMessage(content="Hi! I'm DineMate, here to help you find great restaurants! 👋")]}
    
    # Generate smalltalk response
    smalltalk_messages = [
        SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
        latest_user_msg
    ]
    
    response = llm.invoke(smalltalk_messages)
    
    print(f"✅ [SMALLTALK] Generated response")
    
    return {"messages": [AIMessage(content=response.content.strip())]}


# ========================================
# Graph Routing
# ========================================

def route_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "smalltalk")
    
    print(f"🔀 [ROUTER] Routing based on intent: {intent}")
    
    if intent == "smalltalk":
        return "smalltalk"
    else:
        # Restaurant search or info question - go through the full pipeline
        return "search"


# ========================================
# Graph Construction
# ========================================

def create_v2_graph():
    """Create and compile the v2 LangGraph agent with Cerebras category selector."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classifier", classifier_node)
    graph.add_node("category_selector", category_selector_node)  # Cerebras-powered
    graph.add_node("slot_filler", slot_filler_node)
    graph.add_node("tool_runner", tool_runner_node)
    graph.add_node("response", response_node)
    graph.add_node("smalltalk", smalltalk_responder_node)
    
    # Add edges
    graph.add_edge(START, "classifier")
    
    # Conditional routing from classifier
    graph.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "smalltalk": "smalltalk",
            "search": "category_selector"  # Go to Cerebras category selector first
        }
    )
    
    # Smalltalk path
    graph.add_edge("smalltalk", END)
    
    # Search path (with Cerebras optimization)
    graph.add_edge("category_selector", "slot_filler")  # Cerebras → Gemini
    graph.add_edge("slot_filler", "tool_runner")
    graph.add_edge("tool_runner", "response")
    graph.add_edge("response", END)
    
    return graph.compile()


# Compile the graph
app_v2 = create_v2_graph()


# ========================================
# Main Entrypoint
# ========================================

def chat_with_bot_v2(
    user_input: str,
    memory: Optional[ConversationMemory] = None,
    user_id: Optional[str] = None,
    authenticated_client: Optional[Client] = None,
    current_user: Optional[dict] = None,
) -> str:
    """
    Chat with the v2 bot using structured multi-stage pipeline.
    
    Args:
        user_input: User's message
        memory: Optional conversation memory for context
        user_id: Optional user ID for profile personalization
        authenticated_client: Optional authenticated Supabase client
        current_user: Optional current user data
    
    Returns:
        AI response text with RESTAURANTS_TO_SHOW line if applicable
    """
    try:
        print("\n" + "="*60)
        print("🚀 DineMate v2 - Starting chat processing")
        print("="*60)
        
        # Set up thread-local Supabase client
        if authenticated_client:
            threading.current_thread().supabase_client = authenticated_client
        
        # Log authentication status
        if current_user:
            print(f"👤 Authenticated user: {current_user.get('email', 'unknown')}")
        else:
            print("👤 Unauthenticated user")
        
        # Fetch user profile if available
        user_profile = None
        if user_id:
            client_to_use = authenticated_client if authenticated_client else get_supabase_client()
            user_profile = fetch_user_profile(user_id, client_to_use)
            if user_profile:
                print(f"✅ Loaded user profile: {user_profile.get('full_name', 'Unknown')}")
        
        # Fetch cuisine types for context (same as v1)
        cuisines_message = None
        try:
            cuisines_raw = getAllCuisineTypes()
            cuisines_list = json.loads(cuisines_raw) if cuisines_raw else []
            if isinstance(cuisines_list, list) and cuisines_list:
                try:
                    cuisines_list = sorted(set(map(str, cuisines_list)), key=lambda x: x.lower())
                except Exception:
                    cuisines_list = list(dict.fromkeys(map(str, cuisines_list)))
                cuisines_message = SystemMessage(content="AVAILABLE_CUISINES: " + ", ".join(cuisines_list))
        except Exception:
            cuisines_message = None
        
        # Build initial messages
        messages_to_add = []
        if cuisines_message:
            messages_to_add.append(cuisines_message)
        messages_to_add.append(HumanMessage(content=user_input))
        
        # Handle conversation memory
        if memory:
            history_messages = memory.get_messages()
            initial_messages = history_messages + messages_to_add
        else:
            initial_messages = messages_to_add
        
        # Initialize state
        initial_state: AgentState = {
            "messages": initial_messages,
            "intent": "",
            "search_type": "",
            "wants_restaurants": False,
            "is_refinement": False,
            "selected_categories": [],
            "filters": {},
            "previous_filters": {},
            "restaurants": [],
            "previous_restaurants": [],
            "user_profile": user_profile
        }
        
        # Invoke the v2 graph
        print("⚙️ Invoking v2 graph...")
        result = app_v2.invoke(initial_state)
        
        # Extract final AI response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        
        if ai_messages:
            final_response = ai_messages[-1].content
            
            # Save to memory if provided
            if memory:
                memory.add_message(HumanMessage(content=user_input))
                memory.add_message(ai_messages[-1])
            
            print("="*60)
            print("✅ DineMate v2 - Processing complete")
            print("="*60 + "\n")
            
            return final_response
        else:
            print("⚠️ No AI response generated")
            return "I apologize, I couldn't generate a proper response. Please try again."
    
    except Exception as e:
        print(f"❌ Error in chat_with_bot_v2: {e}")
        import traceback
        traceback.print_exc()
        raise


# ========================================
# Interactive Testing
# ========================================

def start_interactive_chat_v2():
    """Start an interactive chat session with v2 agent for testing."""
    print("🍽️ Welcome to DineMate v2 - Restaurant Assistant (Multi-Stage Pipeline)")
    print("Commands:")
    print("  - Type 'quit' to exit")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'history' to see conversation context")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using DineMate v2! Goodbye! 👋")
            break
        elif user_input.lower() == 'clear':
            memory.clear()
            print("🧹 Conversation history cleared!")
            continue
        elif user_input.lower() == 'history':
            messages = memory.get_messages()
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
            continue
        
        print("DineMate v2: ", end="", flush=True)
        response = chat_with_bot_v2(user_input, memory=memory)
        print(response)


if __name__ == "__main__":
    start_interactive_chat_v2()

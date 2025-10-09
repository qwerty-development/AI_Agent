"""
Restaurant Staff AI Agent with Advanced Table Recommendation System

This module provides an AI assistant specifically designed for restaurant staff operations.
It includes sophisticated table recommendation capabilities using the RMS database's
suggest_optimal_tables PostgreSQL function for intelligent table selection.

Key Features:
- Advanced table recommendations using database-level algorithms
- Real-time availability checking with capacity optimization
- Table combination validation for larger parties
- Hourly availability reporting for operational planning
- Customer history analysis and VIP recognition
- Waitlist management and wait time estimation
- Comprehensive booking and operational statistics

New Advanced Tools:
- getOptimalTableRecommendations: Uses suggest_optimal_tables database function
- validateTableCombination: Validates table combinations using database logic
- getTableAvailabilityReport: Generates hourly availability reports

The agent follows a strict tool-based workflow pattern and maintains conversation
context for enhanced staff interactions.
"""

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
import json
from datetime import datetime, timedelta, date, time as dt_time
from collections import defaultdict

load_dotenv()
url: str = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)

def get_supabase_client() -> Optional[Client]:
    """Get the appropriate Supabase client (authenticated if available, otherwise global)"""
    import threading
    if hasattr(threading.current_thread(), 'supabase_client'):
        return threading.current_thread().supabase_client
    return supabase

class StaffAgentState(TypedDict):
    """State of the restaurant staff agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = []

system_prompt = """
You are an AI assistant specifically designed to help restaurant staff work more efficiently and provide better service. Your ONLY role is to:

1. **Advanced Table Assignment Helper**
   - Use the optimal table recommendation system for intelligent table selection
   - Suggest single tables or combinations based on sophisticated algorithms
   - Consider real-time availability, capacity matching, and priority scoring
   - Validate table combinations before suggesting them to staff
   - Generate hourly availability reports for better planning

2. **Smart Table Assignment Helper (Legacy)**
   - Suggest optimal table assignments based on party size, customer preferences, and current occupancy
   - Recommend table combinations for larger parties
   - Consider customer history and preferences when suggesting tables

2. **Staff Decision Helper**
   - Answer questions about table availability and timing
   - Help with complex booking decisions
   - Suggest solutions for challenging seating situations

3. **Customer Context Assistant**
   - Provide customer insights from booking history
   - Show customer preferences, dietary restrictions, and special notes
   - Help staff provide personalized service

4. **Operations Insights**
   - Answer questions about today's bookings, capacity, and busy times
   - Provide quick stats about restaurant operations
   - Help with planning and resource allocation

5. **Quick Operational Answers**
   - Answer questions about covers, peak times, and table status
   - Provide insights from booking and customer data
   - Help staff make informed decisions quickly

6. **Waitlist Management**
   - View current waitlist and who’s next
   - Summarize waitlist volume and average waits
   - Estimate wait time for a given party size

**IMPORTANT CONSTRAINTS:**
- ONLY answer questions related to restaurant operations, staff assistance, and customer service
- DO NOT provide code, programming solutions, or technical implementations
- DO NOT answer questions outside the scope of restaurant staff assistance
- If asked about non-restaurant topics, politely redirect to restaurant operations
- Always base responses on current restaurant data
- Be concise and actionable - staff need quick, practical answers
- Use tools to gather information first, then call finishedUsingTools ONLY when ready to provide the final response
- Focus on helping staff provide excellent customer service
- NEVER call finishedUsingTools in the same response as data-gathering tools
- Call finishedUsingTools only after you have processed tool results and are ready to give a final answer
- For table recommendations, use BOTH getOptimalTableRecommendations (for smart suggestions) AND getAvailableTables (for complete alternatives) to give staff comprehensive options

**RESPONSE STYLE:**
- Be friendly but professional
- Provide actionable suggestions
- Include relevant details (table numbers, customer names, specific recommendations)
- Use bullet points for multiple suggestions
- Highlight important information (VIP customers, allergies, special occasions)
- ALWAYS provide a clear, helpful response after using tools
- When asked about booking counts, give specific numbers and helpful context
- For table suggestions, explain why specific tables or combinations are recommended
- **FOR TABLE RECOMMENDATIONS: Always show BOTH the optimal recommended tables AND list other available tables as alternatives**, giving staff complete options to choose from

**TABLE RECOMMENDATION FORMAT EXAMPLE:**
"For your party of 4, here are my recommendations:

**🌟 RECOMMENDED TABLES (Optimal choices):**
• Table 12 (4-seat booth) - Perfect size, quiet corner
• Tables 5+6 combined (8 seats) - Popular combination for groups

**📋 OTHER AVAILABLE TABLES:**
• Table 8 (6-seat round) - Slightly larger but available
• Table 15 (2-seat) + Table 16 (2-seat) - Alternative combination
• Table 20 (8-seat) - Large table, good for celebrations"

**WORKFLOW:**
1. Use relevant tools to gather information (e.g., getOptimalTableRecommendations for table suggestions)
2. Wait for tool results
3. Call finishedUsingTools when ready to respond 
4. Provide a natural language response with the information gathered
"""

@tool
def finishedUsingTools() -> str:
    """Call this when you're done using tools. IMPORTANT: After calling this tool, you MUST provide a helpful natural language response to the staff member in the same message."""
    print("Staff AI finished using tools")
    return "Tools usage completed. Now provide a helpful response to the staff member based on the data you gathered. Include specific numbers, details, and actionable information."

tools.append(finishedUsingTools)

@tool
def getTodaysBookings(restaurant_id: str) -> str:
    """Get all bookings for today for a specific restaurant"""
    try:
        today = date.today()
        start_of_day = datetime.combine(today, dt_time.min)
        end_of_day = datetime.combine(today, dt_time.max)
        
        result = get_supabase_client().table("bookings").select("""
            id, user_id, booking_time, party_size, status, special_requests, 
            occasion, dietary_notes, guest_name, guest_email, guest_phone,
            confirmation_code, checked_in_at, seated_at, 
            profiles!bookings_user_id_fkey(full_name, phone_number, allergies, dietary_restrictions)
        """).eq("restaurant_id", restaurant_id).gte("booking_time", start_of_day.isoformat()).lte("booking_time", end_of_day.isoformat()).order("booking_time").execute()
        
        bookings = result.data
        if not bookings:
            return "No bookings found for today"
        
        return json.dumps(bookings)
    except Exception as e:
        return f"Error retrieving today's bookings: {str(e)}"

tools.append(getTodaysBookings)

@tool
def getAvailableTables(restaurant_id: str, desired_time: str = None, party_size: int = None) -> str:
    """Get available tables for a restaurant, optionally filtered by time and party size"""
    print(f"Staff AI is checking available tables for restaurant: {restaurant_id}")
    try:
        # Get all tables for the restaurant
        tables_result = get_supabase_client().table("restaurant_tables").select("""
            id, table_number, table_type, capacity, min_capacity, max_capacity, 
            features, is_active, x_position, y_position
        """).eq("restaurant_id", restaurant_id).eq("is_active", True).order("table_number").execute()
        
        tables = tables_result.data
        if not tables:
            return "No tables found for this restaurant"
        
        # If specific time and party size provided, check availability
        if desired_time and party_size:
            # Get current bookings that might conflict
            booking_time = datetime.fromisoformat(desired_time.replace('Z', '+00:00'))
            start_time = booking_time - timedelta(hours=2)  # 2 hour window before
            end_time = booking_time + timedelta(hours=2)    # 2 hour window after
            
            bookings_result = get_supabase_client().table("bookings").select("""
                id, booking_time, party_size, status, booking_tables(table_id)
            """).eq("restaurant_id", restaurant_id).gte("booking_time", start_time.isoformat()).lte("booking_time", end_time.isoformat()).in_("status", ["confirmed", "seated", "arrived"]).execute()
            
            booked_table_ids = set()
            for booking in bookings_result.data:
                if booking.get("booking_tables"):
                    for bt in booking["booking_tables"]:
                        booked_table_ids.add(bt["table_id"])
            
            # Filter available tables
            available_tables = []
            for table in tables:
                if table["id"] not in booked_table_ids and table["min_capacity"] <= party_size <= table["max_capacity"]:
                    available_tables.append(table)
            
            result_data = {
                "available_tables": available_tables,
                "requested_time": desired_time,
                "requested_party_size": party_size,
                "total_tables": len(tables),
                "available_count": len(available_tables)
            }
        else:
            result_data = {
                "all_tables": tables,
                "total_tables": len(tables)
            }
        
        return json.dumps(result_data)
    except Exception as e:
        print(f"Error fetching available tables: {e}")
        return f"Error retrieving available tables: {str(e)}"

tools.append(getAvailableTables)

@tool
def getCustomerHistory(customer_identifier: str, restaurant_id: str) -> str:
    """Get customer booking history and preferences. Use email, phone, or name to identify customer"""
    print(f"Staff AI is looking up customer history: {customer_identifier}")
    try:
        # First try to find customer by different identifiers
        customer_query = get_supabase_client().table("restaurant_customers").select("""
            id, user_id, guest_email, guest_phone, guest_name, total_bookings,
            total_spent, average_party_size, last_visit, first_visit, 
            no_show_count, cancelled_count, vip_status, preferred_table_types,
            preferred_time_slots, 
            profiles!restaurant_customers_user_id_fkey(full_name, phone_number, allergies, dietary_restrictions)
        """).eq("restaurant_id", restaurant_id)
        
        # Try different ways to match the customer
        if "@" in customer_identifier:
            customer_query = customer_query.eq("guest_email", customer_identifier)
        elif customer_identifier.isdigit():
            customer_query = customer_query.eq("guest_phone", customer_identifier)
        else:
            customer_query = customer_query.ilike("guest_name", f"%{customer_identifier}%")
        
        customer_result = customer_query.execute()
        customers = customer_result.data
        
        if not customers:
            return f"No customer found with identifier: {customer_identifier}"
        
        customer = customers[0]  # Take first match
        
        # Get recent bookings for this customer
        recent_bookings = get_supabase_client().table("bookings").select("""
            id, booking_time, party_size, status, special_requests, occasion,
            dietary_notes, confirmation_code
        """).eq("restaurant_id", restaurant_id).or_(
            f"user_id.eq.{customer.get('user_id', 'null')},guest_email.eq.{customer.get('guest_email', 'null')}"
        ).order("booking_time", desc=True).limit(5).execute()
        
        # Get customer notes if any
        notes_result = get_supabase_client().table("customer_notes").select("""
            note, category, is_important, created_at
        """).eq("customer_id", customer["id"]).order("created_at", desc=True).execute()
        
        customer_data = {
            "customer_info": customer,
            "recent_bookings": recent_bookings.data,
            "customer_notes": notes_result.data,
            "summary": {
                "total_visits": customer.get("total_bookings", 0),
                "vip_status": customer.get("vip_status", False),
                "no_shows": customer.get("no_show_count", 0),
                "cancellations": customer.get("cancelled_count", 0),
                "last_visit": customer.get("last_visit"),
                "average_party": customer.get("average_party_size", 0)
            }
        }
        
        return json.dumps(customer_data)
    except Exception as e:
        print(f"Error fetching customer history: {e}")
        return f"Error retrieving customer history: {str(e)}"

tools.append(getCustomerHistory)

@tool
def getTableSuggestions(restaurant_id: str, party_size: int, customer_preferences: str = None, booking_time: str = None) -> str:
    """Get smart table suggestions based on party size and preferences"""
    print(f"Staff AI is suggesting tables for party of {party_size}")
    try:
        # Get available tables
        tables_result = get_supabase_client().table("restaurant_tables").select("""
            id, table_number, table_type, capacity, min_capacity, max_capacity,
            features, x_position, y_position, priority_score
        """).eq("restaurant_id", restaurant_id).eq("is_active", True).execute()
        
        tables = tables_result.data
        if not tables:
            return "No tables available"
        
        # Score tables based on suitability
        suggestions = []
        for table in tables:
            if table["min_capacity"] <= party_size <= table["max_capacity"]:
                score = 0
                reasons = []
                
                # Perfect size match gets higher score
                if table["capacity"] == party_size:
                    score += 10
                    reasons.append("Perfect size match")
                elif table["capacity"] == party_size + 1:
                    score += 8
                    reasons.append("Optimal size")
                
                # Table type preferences
                if customer_preferences:
                    prefs_lower = customer_preferences.lower()
                    if "window" in prefs_lower and table["table_type"] == "window":
                        score += 5
                        reasons.append("Window seating preference")
                    elif "booth" in prefs_lower and table["table_type"] == "booth":
                        score += 5
                        reasons.append("Booth seating preference")
                    elif "patio" in prefs_lower and table["table_type"] == "patio":
                        score += 5
                        reasons.append("Outdoor seating preference")
                
                # Features bonus
                features = table.get("features", [])
                if features:
                    score += 2
                    reasons.append(f"Special features: {', '.join(features)}")
                
                # Priority score from restaurant
                score += table.get("priority_score", 0)
                
                suggestions.append({
                    "table": table,
                    "score": score,
                    "reasons": reasons,
                    "recommendation": f"Table {table['table_number']} ({table['table_type']}, seats {table['capacity']})"
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        # Also check for table combinations if no perfect match
        combinations = []
        if party_size > max(table["max_capacity"] for table in tables):
            combo_result = get_supabase_client().table("table_combinations").select("""
                id, primary_table_id, secondary_table_id, combined_capacity,
                restaurant_tables!primary_table_id(table_number, table_type),
                restaurant_tables!secondary_table_id(table_number, table_type)
            """).eq("restaurant_id", restaurant_id).eq("is_active", True).execute()
            
            for combo in combo_result.data:
                if combo["combined_capacity"] >= party_size:
                    combinations.append(combo)
        
        result = {
            "party_size": party_size,
            "individual_suggestions": suggestions[:5],  # Top 5
            "combination_options": combinations,
            "total_suitable_tables": len(suggestions)
        }
        
        return json.dumps(result)
    except Exception as e:
        print(f"Error getting table suggestions: {e}")
        return f"Error getting table suggestions: {str(e)}"

tools.append(getTableSuggestions)

@tool
def getRestaurantStats(restaurant_id: str, date_filter: str = "today") -> str:
    """Get restaurant operational statistics for today, week, or month"""
    print(f"Staff AI is getting restaurant stats for: {date_filter}")
    try:
        today = date.today()
        
        if date_filter == "today":
            start_date = datetime.combine(today, dt_time.min)
            end_date = datetime.combine(today, dt_time.max)
        elif date_filter == "week":
            start_date = datetime.combine(today - timedelta(days=7), dt_time.min)
            end_date = datetime.combine(today, dt_time.max)
        elif date_filter == "month":
            start_date = datetime.combine(today - timedelta(days=30), dt_time.min)
            end_date = datetime.combine(today, dt_time.max)
        else:
            start_date = datetime.combine(today, dt_time.min)
            end_date = datetime.combine(today, dt_time.max)
        
        # Get booking statistics
        bookings_result = get_supabase_client().table("bookings").select("""
            id, booking_time, party_size, status, created_at
        """).eq("restaurant_id", restaurant_id).gte("booking_time", start_date.isoformat()).lte("booking_time", end_date.isoformat()).execute()
        
        bookings = bookings_result.data
        
        # Calculate stats
        total_bookings = len(bookings)
        total_covers = sum(booking["party_size"] for booking in bookings)
        
        status_counts = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for booking in bookings:
            status_counts[booking["status"]] += 1
            
            # Hour distribution
            booking_hour = datetime.fromisoformat(booking["booking_time"].replace('Z', '+00:00')).hour
            hourly_distribution[booking_hour] += booking["party_size"]
        
        # Find peak hours
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else (None, 0)
        
        # Calculate rates
        confirmed_bookings = status_counts.get("confirmed", 0) + status_counts.get("seated", 0) + status_counts.get("completed", 0)
        no_shows = status_counts.get("no_show", 0)
        cancellations = status_counts.get("cancelled_by_user", 0) + status_counts.get("cancelled_by_restaurant", 0)
        
        stats = {
            "period": date_filter,
            "total_bookings": total_bookings,
            "total_covers": total_covers,
            "confirmed_bookings": confirmed_bookings,
            "no_shows": no_shows,
            "cancellations": cancellations,
            "status_breakdown": dict(status_counts),
            "peak_hour": {
                "hour": peak_hour[0],
                "covers": peak_hour[1]
            } if peak_hour[0] is not None else None,
            "hourly_covers": dict(hourly_distribution),
            "average_party_size": round(total_covers / total_bookings, 1) if total_bookings > 0 else 0,
            "no_show_rate": round((no_shows / total_bookings * 100), 1) if total_bookings > 0 else 0
        }
        
        return json.dumps(stats)
    except Exception as e:
        print(f"Error getting restaurant stats: {e}")
        return f"Error getting restaurant stats: {str(e)}"

tools.append(getRestaurantStats)

@tool
def checkBookingDetails(confirmation_code: str = None, booking_id: str = None) -> str:
    """Get detailed information about a specific booking using confirmation code or booking ID"""
    print(f"Staff AI is checking booking details")
    try:
        query = get_supabase_client().table("bookings").select("""
            id, user_id, booking_time, party_size, status, special_requests,
            occasion, dietary_notes, guest_name, guest_email, guest_phone,
            confirmation_code, checked_in_at, seated_at, turn_time_minutes,
            booking_tables(restaurant_tables(table_number, table_type, capacity, features)),
            profiles!bookings_user_id_fkey(full_name, phone_number, allergies, dietary_restrictions)
        """)
        
        if confirmation_code:
            query = query.eq("confirmation_code", confirmation_code)
        elif booking_id:
            query = query.eq("id", booking_id)
        else:
            return "Please provide either confirmation code or booking ID"
        
        result = query.execute()
        bookings = result.data
        
        if not bookings:
            return "No booking found with the provided details"
        
        booking = bookings[0]
        
        # Get customer notes if user_id exists
        customer_notes = []
        if booking.get("user_id"):
            notes_result = get_supabase_client().table("customer_notes").select("""
                note, category, is_important, created_at
            """).eq("customer_id", booking["user_id"]).order("created_at", desc=True).limit(3).execute()
            customer_notes = notes_result.data
        
        booking_details = {
            "booking_info": booking,
            "customer_notes": customer_notes,
            "assigned_tables": [table["restaurant_tables"] for table in booking.get("booking_tables", [])]
        }
        
        return json.dumps(booking_details)
    except Exception as e:
        print(f"Error checking booking details: {e}")
        return f"Error checking booking details: {str(e)}"

tools.append(checkBookingDetails)

# -----------------------------
# Waitlist tools (read/insights)
# -----------------------------

@tool
def getWaitlist(restaurant_id: str, status: str = None) -> str:
    """Get current waitlist entries for a restaurant. Optionally filter by status (e.g., 'waiting')."""
    print(f"Staff AI is fetching waitlist for restaurant: {restaurant_id}")
    try:
        # Select known, schema-friendly columns
        query = (
            supabase
            .table("waitlist")
            .select("id, restaurant_id, party_size, status, joined_at, quoted_wait_minutes, priority, notified_at")
            .eq("restaurant_id", restaurant_id)
        )
        if status:
            query = query.eq("status", status)
        try:
            result = query.order("joined_at").execute()
        except Exception:
            # Fallback to all columns if specific list fails
            result = (
                supabase
                .table("waitlist")
                .select("*")
                .eq("restaurant_id", restaurant_id)
                .execute()
            )

        entries = result.data
        if not entries:
            return json.dumps({"count": 0, "entries": []})

        return json.dumps({"count": len(entries), "entries": entries})
    except Exception as e:
        print(f"Error fetching waitlist: {e}")
        return f"Error retrieving waitlist: {str(e)}"

tools.append(getWaitlist)

@tool
def getWaitlistStats(restaurant_id: str) -> str:
    """Get summary stats for the restaurant waitlist: counts by status, average quoted wait, next in line."""
    print(f"Staff AI is computing waitlist stats for restaurant: {restaurant_id}")
    try:
        # Fetch all current entries
        try:
            result = (
                supabase
                .table("waitlist")
                .select("id, restaurant_id, party_size, status, joined_at, quoted_wait_minutes, priority, notified_at")
                .eq("restaurant_id", restaurant_id)
                .execute()
            )
        except Exception:
            result = get_supabase_client().table("waitlist").select("*").eq("restaurant_id", restaurant_id).execute()

        entries = result.data or []

        # Aggregate
        status_counts = defaultdict(int)
        quoted_minutes = []
        waiting_entries = []
        for entry in entries:
            status_value = entry.get("status", "unknown")
            status_counts[status_value] += 1
            qm = entry.get("quoted_wait_minutes")
            if isinstance(qm, (int, float)):
                quoted_minutes.append(qm)
            if status_value in ("waiting", "notified", "queued", "pending"):
                waiting_entries.append(entry)

        # Sort waiting by priority then joined_at
        def parse_joined_at(value: str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00')) if value else datetime.min
            except Exception:
                return datetime.min

        waiting_entries.sort(key=lambda e: (
            -(e.get("priority") or 0),
            parse_joined_at(e.get("joined_at"))
        ))

        average_quoted = round(sum(quoted_minutes) / len(quoted_minutes), 1) if quoted_minutes else None

        next_up = waiting_entries[0] if waiting_entries else None

        stats = {
            "total_waiting": sum(status_counts.get(s, 0) for s in ["waiting", "notified", "queued", "pending"]),
            "status_breakdown": dict(status_counts),
            "average_quoted_wait_minutes": average_quoted,
            "next_up": next_up,
        }

        return json.dumps(stats)
    except Exception as e:
        print(f"Error computing waitlist stats: {e}")
        return f"Error computing waitlist stats: {str(e)}"

tools.append(getWaitlistStats)

@tool
def estimateWaitTime(restaurant_id: str, party_size: int) -> str:
    """Estimate wait time (minutes) for a given party size using current waitlist data."""
    print(f"Staff AI is estimating wait time for party of {party_size} at restaurant: {restaurant_id}")
    try:
        # Load waitlist
        try:
            wl_result = (
                supabase
                .table("waitlist")
                .select("id, party_size, status, joined_at, quoted_wait_minutes, priority")
                .eq("restaurant_id", restaurant_id)
                .execute()
            )
        except Exception:
            wl_result = get_supabase_client().table("waitlist").select("*").eq("restaurant_id", restaurant_id).execute()

        entries = wl_result.data or []

        # Consider only active/waiting entries
        active_statuses = {"waiting", "notified", "queued", "pending"}
        waiting_entries = [e for e in entries if (e.get("status") in active_statuses)]

        # If there are quoted waits for similar party sizes, use their average
        similar_quotes = [e.get("quoted_wait_minutes") for e in waiting_entries if isinstance(e.get("quoted_wait_minutes"), (int, float)) and abs((e.get("party_size") or 0) - party_size) <= 1]
        if similar_quotes:
            estimate = round(sum(similar_quotes) / len(similar_quotes))
        else:
            # Fallback heuristic: 12 minutes per party ahead (light load), 18 if high load
            parties_ahead = sum(1 for e in waiting_entries if (e.get("party_size") or 0) <= party_size)
            per_party_min = 12 if len(waiting_entries) <= 5 else 18
            estimate = max(per_party_min, parties_ahead * per_party_min)

        return json.dumps({
            "party_size": party_size,
            "estimated_wait_minutes": int(estimate),
            "active_waitlist_count": len(waiting_entries)
        })
    except Exception as e:
        print(f"Error estimating wait time: {e}")
        return f"Error estimating wait time: {str(e)}"

tools.append(estimateWaitTime)

@tool
def getOptimalTableRecommendations(restaurant_id: str, party_size: int, booking_time: str = "now", turn_time_minutes: int = 120) -> str:
    """
    Get optimal table recommendations using the advanced RMS algorithm. 
    This uses the suggest_optimal_tables database function for intelligent table selection.
    booking_time can be:
    - "now" or "current" for current time
    - "19:00" for today at 7 PM 
    - Full ISO format "2024-08-27T19:00:00"
    """
    print(f"Staff AI is getting optimal table recommendations for party of {party_size}")
    try:
        # Handle different time formats
        if booking_time.lower() in ["now", "current", "right now"]:
            start_time = datetime.now()
        elif ":" in booking_time and "T" not in booking_time and len(booking_time) <= 5:
            # Handle "19:00" format - assume today
            today = date.today()
            time_part = booking_time.strip()
            if len(time_part.split(':')) == 2:
                hour, minute = map(int, time_part.split(':'))
                start_time = datetime.combine(today, dt_time(hour, minute))
            else:
                raise ValueError(f"Invalid time format: {booking_time}")
        else:
            # Handle full ISO format or other standard formats
            start_time = datetime.fromisoformat(booking_time.replace('Z', '+00:00'))
        
        end_time = start_time + timedelta(minutes=turn_time_minutes)
        
        # Call the suggest_optimal_tables database function
        result = get_supabase_client().rpc('suggest_optimal_tables', {
            'p_restaurant_id': restaurant_id,
            'p_party_size': party_size,
            'p_start_time': start_time.isoformat(),
            'p_end_time': end_time.isoformat()
        }).execute()
        
        recommendations = result.data
        
        if not recommendations:
            return json.dumps({
                "status": "no_availability",
                "message": "No suitable tables available for the requested time and party size",
                "party_size": party_size,
                "requested_time": booking_time
            })
        
        # Get detailed table information for the recommended tables
        recommendation = recommendations[0]  # Take the first (best) recommendation
        table_ids = recommendation.get('table_ids', [])
        total_capacity = recommendation.get('total_capacity', 0)
        requires_combination = recommendation.get('requires_combination', False)
        
        # Fetch detailed table information
        tables_info = []
        if table_ids:
            tables_result = get_supabase_client().table("restaurant_tables").select("""
                id, table_number, table_type, capacity, min_capacity, max_capacity,
                features, x_position, y_position, priority_score
            """).in_("id", table_ids).execute()
            
            tables_info = tables_result.data
        
        # Format the recommendation response
        recommendation_data = {
            "status": "success",
            "party_size": party_size,
            "requested_time": booking_time,
            "total_capacity": total_capacity,
            "requires_combination": requires_combination,
            "recommended_tables": tables_info,
            "table_count": len(table_ids),
            "algorithm_notes": {
                "selection_method": "combination" if requires_combination else "single_table",
                "optimization": "closest_capacity_match_with_priority_scoring"
            }
        }
        
        return json.dumps(recommendation_data)
        
    except Exception as e:
        print(f"Error getting optimal table recommendations: {e}")
        return f"Error getting optimal table recommendations: {str(e)}"

tools.append(getOptimalTableRecommendations)

@tool
def getTableCombinationsNow(restaurant_id: str, party_size: int) -> str:
    """
    Get table combination recommendations for a party right now.
    Specifically designed for immediate seating needs when staff say "right now" or "current".
    """
    print(f"Staff AI is getting immediate table combinations for party of {party_size}")
    try:
        # Use current time
        current_time = datetime.now()
        
        # Call the suggest_optimal_tables database function
        result = get_supabase_client().rpc('suggest_optimal_tables', {
            'p_restaurant_id': restaurant_id,
            'p_party_size': party_size,
            'p_start_time': current_time.isoformat(),
            'p_end_time': (current_time + timedelta(hours=2)).isoformat()
        }).execute()
        
        recommendations = result.data
        
        if not recommendations:
            return json.dumps({
                "status": "no_availability",
                "message": f"No suitable table combinations available right now for party of {party_size}",
                "party_size": party_size,
                "current_time": current_time.strftime("%H:%M")
            })
        
        # Get detailed table information for the recommended tables
        recommendation = recommendations[0]  # Take the first (best) recommendation
        table_ids = recommendation.get('table_ids', [])
        total_capacity = recommendation.get('total_capacity', 0)
        requires_combination = recommendation.get('requires_combination', False)
        
        # Fetch detailed table information
        tables_info = []
        if table_ids:
            tables_result = get_supabase_client().table("restaurant_tables").select("""
                id, table_number, table_type, capacity, min_capacity, max_capacity,
                features, x_position, y_position, priority_score
            """).in_("id", table_ids).execute()
            
            tables_info = tables_result.data
        
        # Format the response specifically for immediate table combinations
        response_data = {
            "status": "success",
            "party_size": party_size,
            "current_time": current_time.strftime("%H:%M"),
            "total_capacity": total_capacity,
            "requires_combination": requires_combination,
            "recommended_tables": tables_info,
            "table_count": len(table_ids),
            "immediate_setup": True,
            "setup_instructions": f"Combine {len(table_ids)} tables" if requires_combination else f"Use single table {tables_info[0].get('table_number', '?') if tables_info else '?'}"
        }
        
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"Error getting immediate table combinations: {e}")
        return f"Error getting immediate table combinations: {str(e)}"

tools.append(getTableCombinationsNow)

@tool 
def validateTableCombination(restaurant_id: str, table_ids: str, party_size: int) -> str:
    """
    Validate if a specific combination of tables is suitable for a party size.
    Uses the validate_table_combination database function.
    table_ids should be a comma-separated string of table IDs.
    """
    print(f"Staff AI is validating table combination: {table_ids}")
    try:
        # Convert comma-separated string to list
        table_id_list = [id.strip() for id in table_ids.split(',') if id.strip()]
        
        # Call the validate_table_combination database function
        result = get_supabase_client().rpc('validate_table_combination', {
            'p_table_ids': table_id_list,
            'p_party_size': party_size
        }).execute()
        
        validation_result = result.data[0] if result.data else {}
        
        validation_data = {
            "is_valid": validation_result.get('is_valid', False),
            "total_capacity": validation_result.get('total_capacity', 0),
            "validation_message": validation_result.get('validation_message', ''),
            "table_ids": table_id_list,
            "table_ids_input": table_ids,
            "party_size": party_size
        }
        
        return json.dumps(validation_data)
        
    except Exception as e:
        print(f"Error validating table combination: {e}")
        return f"Error validating table combination: {str(e)}"

tools.append(validateTableCombination)

@tool
def getTableAvailabilityReport(restaurant_id: str, date: str) -> str:
    """
    Get hourly table availability report for a specific date.
    Uses the get_table_availability_by_hour database function.
    """
    print(f"Staff AI is generating table availability report for {date}")
    try:
        # Call the get_table_availability_by_hour database function
        result = get_supabase_client().rpc('get_table_availability_by_hour', {
            'p_restaurant_id': restaurant_id,
            'p_date': date
        }).execute()
        
        hourly_data = result.data or []
        
        # Format the report
        report = {
            "restaurant_id": restaurant_id,
            "date": date,
            "hourly_availability": hourly_data,
            "summary": {
                "peak_hours": [h for h in hourly_data if h.get('utilization_percentage', 0) > 80],
                "quiet_hours": [h for h in hourly_data if h.get('utilization_percentage', 0) < 30],
                "total_hours_analyzed": len(hourly_data)
            }
        }
        
        return json.dumps(report)
        
    except Exception as e:
        print(f"Error generating availability report: {e}")
        return f"Error generating availability report: {str(e)}"

tools.append(getTableAvailabilityReport)

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)
llm = llm.bind_tools(tools)

def staff_agent_node(state: StaffAgentState) -> StaffAgentState:
    """Our staff agent node that processes messages and generates responses."""
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

    print(f"Sending {len(full_messages)} messages to Staff LLM (system_prompt_included: {not has_system_message})")

    # Get response from the model
    response = llm.invoke(full_messages)
    
    # Return the updated state with the new message
    return {"messages": [response]}

def should_continue(state: StaffAgentState) -> str:
    """Determine whether to continue with tools or end the conversation"""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, 'tool_calls', []) or []
        print(f"Staff AI tool calls found: {len(tool_calls)}")
        
        # If there are tool calls, check if finishedUsingTools was called
        for call in tool_calls:
            print(f"Staff AI tool call: {call}")
            if call["name"] == "finishedUsingTools":
                print("✅ Staff AI called finishedUsingTools tool - executing tool and continuing")
                # Execute the tool so the agent can produce a final response after
                return "continue"
        
        # If there are other tool calls, continue to tools
        if tool_calls:
            print("🔧 Staff AI has tool calls - continuing to tools")
            return "continue"
        
        # If no tool calls and has content, end
        if last_message.content:
            print("💬 Staff AI has content but no tool calls - ending")
            return "end"
    
    print("🔄 Staff AI default case - continuing")
    return "continue"

# Create the graph
staff_graph = StateGraph(StaffAgentState)

# Add nodes
staff_graph.add_node("staff_agent", staff_agent_node)
staff_graph.add_node("tools", ToolNode(tools))

# Add edges
staff_graph.add_edge(START, "staff_agent")
staff_graph.add_conditional_edges(
    "staff_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
staff_graph.add_edge("tools", "staff_agent")

# Compile the graph
staff_app = staff_graph.compile()

# Conversation memory system for staff agent
class StaffConversationMemory:
    def __init__(self, max_history: int = 6):
        """Initialize conversation memory with a maximum history limit.
        Reduced from 20 to 6 messages to optimize token usage and reduce costs.
        6 messages = 3 turns of conversation (user + assistant pairs), which is sufficient for context.
        """
        self.messages = []
        self.max_history = max_history
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.messages.append(message)
        # Keep only the most recent messages to prevent context overflow
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_messages(self):
        """Get all messages in the conversation history."""
        return self.messages.copy()
    
    def clear(self):
        """Clear the conversation history."""
        self.messages.clear()
    
    def get_context_size(self) -> int:
        """Get the number of messages in history."""
        return len(self.messages)

def create_staff_conversation_memory(max_history: int = 6):
    """Create a new conversation memory instance for staff agent.
    Default reduced to 6 messages for cost optimization."""
    return StaffConversationMemory(max_history)

def chat_with_staff_bot(user_input: str, restaurant_id: str = None, memory=None, authenticated_client=None, current_user=None) -> str:
    """
    Function to chat with the restaurant staff bot. 
    Supports conversation memory for contextual responses.
    Now supports authenticated Supabase client for RLS compliance.
    """
    try:
        # Store original global client and replace with authenticated one if provided
        global supabase
        original_client = supabase
        if authenticated_client:
            supabase = authenticated_client
        
        try:
            # Enhance the user input with restaurant context if provided
            if restaurant_id:
                enhanced_input = f"[Restaurant ID: {restaurant_id}] {user_input}"
            else:
                enhanced_input = user_input
            
            # Create user message
            user_message = HumanMessage(content=enhanced_input)
            
            # Build message list based on whether we have conversation memory
            if memory:
                # Use conversation history
                history_messages = memory.get_messages()
                current_input = {"messages": history_messages + [user_message]}
            else:
                # Stateless mode - just the current message
                current_input = {"messages": [user_message]}
            
            # Run the staff agent
            result = staff_app.invoke(current_input)
            
            # Look for AI messages and tool results
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
            
            print(f"Found {len(ai_messages)} AI messages and {len(tool_messages)} tool messages")
            
            # If we have tool results but no proper AI response, create one
            if tool_messages and ai_messages:
                last_ai_message = ai_messages[-1]
                
                # Check if the last AI message has content
                if last_ai_message.content and last_ai_message.content.strip():
                    final_response = last_ai_message.content
                    
                    # Save conversation to memory if provided
                    if memory:
                        memory.add_message(user_message)
                        memory.add_message(last_ai_message)
                    
                    return final_response
            
            # If no content but we have tool results, generate a response based on tool data
            # Collect tool outputs by name
            tool_results_by_name = {}
            for tool_msg in tool_messages:
                tool_name = getattr(tool_msg, 'name', None)
                if tool_name:
                    tool_results_by_name[tool_name] = tool_msg.content

            # 1) Today's bookings summary
            booking_tool_result = tool_results_by_name.get('getTodaysBookings')
            if booking_tool_result:
                if booking_tool_result == "No bookings found for today":
                    return "No bookings scheduled for today."
                try:
                    bookings = json.loads(booking_tool_result)
                    if isinstance(bookings, list):
                        count = len(bookings)
                        response = f"You have {count} booking{'s' if count != 1 else ''} for today:\n\n"
                        for i, booking in enumerate(bookings, 1):
                            time = booking.get('booking_time', '').split('T')[1][:5] if 'T' in booking.get('booking_time', '') else 'Unknown time'
                            guest = booking.get('guest_name', 'Unknown guest')
                            party = booking.get('party_size', 'Unknown size')
                            status = booking.get('status', 'Unknown status')
                            response += f"{i}. {guest} - {party} people at {time} ({status})\n"
                        return response.strip()
                except Exception:
                    return "You have bookings for today, but I couldn't parse the details properly."

            # 2) Available tables summary
            available_tables_result = tool_results_by_name.get('getAvailableTables')
            if available_tables_result:
                try:
                    data = json.loads(available_tables_result)
                    if 'available_tables' in data:
                        available = data.get('available_tables', [])
                        count = len(available)
                        requested_time = data.get('requested_time')
                        party_size = data.get('requested_party_size')
                        header = f"Found {count} suitable table{'s' if count != 1 else ''}"
                        if party_size:
                            header += f" for party of {party_size}"
                        if requested_time:
                            header += f" at {requested_time}"
                        header += ":\n"
                        details = []
                        for t in available[:5]:
                            details.append(f"- Table {t.get('table_number', '?')} ({t.get('table_type', 'standard')}), seats {t.get('capacity', '?')}")
                        return (header + ("\n".join(details) if details else "")).strip()
                    elif 'all_tables' in data:
                        total = data.get('total_tables', 0)
                        sample = data.get('all_tables', [])[:5]
                        lines = [f"There are {total} active tables:"]
                        for t in sample:
                            lines.append(f"- Table {t.get('table_number', '?')} ({t.get('table_type', 'standard')}), capacity {t.get('min_capacity', '?')}-{t.get('max_capacity', '?')}")
                        return "\n".join(lines).strip()
                except Exception:
                    pass

            # 3) Optimal table recommendations (new advanced system)
            optimal_rec_result = tool_results_by_name.get('getOptimalTableRecommendations')
            if optimal_rec_result:
                try:
                    data = json.loads(optimal_rec_result)
                    if data.get('status') == 'success':
                        party = data.get('party_size')
                        tables = data.get('recommended_tables', [])
                        combo = data.get('requires_combination', False)
                        capacity = data.get('total_capacity', 0)
                        
                        combo_text = " (table combination)" if combo else ""
                        lines = [f"Optimal recommendation for party of {party}{combo_text}:"]
                        for t in tables:
                            lines.append(f"- Table {t.get('table_number', '?')} ({t.get('table_type', 'standard')}, seats {t.get('capacity', '?')})")
                        lines.append(f"Total capacity: {capacity}")
                        return "\n".join(lines).strip()
                    else:
                        return data.get('message', 'No suitable tables available')
                except Exception:
                    pass

            # 3.5) Immediate table combinations (for "right now" requests)
            immediate_combo_result = tool_results_by_name.get('getTableCombinationsNow')
            if immediate_combo_result:
                try:
                    data = json.loads(immediate_combo_result)
                    if data.get('status') == 'success':
                        party = data.get('party_size')
                        tables = data.get('recommended_tables', [])
                        combo = data.get('requires_combination', False)
                        capacity = data.get('total_capacity', 0)
                        setup = data.get('setup_instructions', '')
                        
                        combo_text = " (table combination needed)" if combo else ""
                        lines = [f"🔄 Immediate seating for party of {party}{combo_text}:"]
                        for t in tables:
                            lines.append(f"- Table {t.get('table_number', '?')} ({t.get('table_type', 'standard')}, seats {t.get('capacity', '?')})")
                        lines.append(f"Total capacity: {capacity}")
                        if setup:
                            lines.append(f"Setup: {setup}")
                        return "\n".join(lines).strip()
                    else:
                        return data.get('message', 'No suitable table combinations available right now')
                except Exception:
                    pass

            # 4) Table suggestions (legacy system)
            suggestions_result = tool_results_by_name.get('getTableSuggestions')
            if suggestions_result:
                try:
                    data = json.loads(suggestions_result)
                    party = data.get('party_size')
                    suggestions = data.get('individual_suggestions', [])
                    lines = [f"Top table suggestions for party of {party}:"]
                    for s in suggestions[:5]:
                        rec = s.get('recommendation')
                        if rec:
                            lines.append(f"- {rec}")
                    return "\n".join(lines).strip()
                except Exception:
                    pass

            # 5) Customer history
            history_result = tool_results_by_name.get('getCustomerHistory')
            if history_result:
                try:
                    data = json.loads(history_result)
                    info = data.get('summary', {})
                    total_visits = info.get('total_visits', 0)
                    vip = info.get('vip_status', False)
                    avg_party = info.get('average_party', 0)
                    return f"Customer summary: {total_visits} total visits, average party size {avg_party}. VIP: {'Yes' if vip else 'No'}."
                except Exception:
                    pass

            # 6) Booking details
            booking_details_result = tool_results_by_name.get('checkBookingDetails')
            if booking_details_result:
                try:
                    data = json.loads(booking_details_result)
                    booking = data.get('booking_info', {})
                    guest = booking.get('guest_name', 'Unknown guest')
                    party = booking.get('party_size', 'Unknown')
                    time = booking.get('booking_time', 'Unknown time')
                    status = booking.get('status', 'Unknown status')
                    return f"Booking details: {guest}, party of {party} at {time} ({status})."
                except Exception:
                    pass

            # 7) Restaurant stats
            stats_result = tool_results_by_name.get('getRestaurantStats')
            if stats_result:
                try:
                    stats = json.loads(stats_result)
                    total = stats.get('total_bookings', 0)
                    covers = stats.get('total_covers', 0)
                    peak = stats.get('peak_hour')
                    peak_text = f", peak hour {peak.get('hour')}:00 with {peak.get('covers')} covers" if peak else ""
                    return f"Today's stats: {total} bookings, {covers} covers{peak_text}."
                except Exception:
                    pass

            # 8) Waitlist entries
            waitlist_result = tool_results_by_name.get('getWaitlist')
            if waitlist_result:
                try:
                    data = json.loads(waitlist_result)
                    count = data.get('count', 0)
                    entries = data.get('entries', [])
                    if count == 0:
                        return "No one is currently on the waitlist."
                    first = entries[0] if entries else {}
                    size = first.get('party_size', '?')
                    return f"Waitlist: {count} part{'ies' if count != 1 else 'y'} waiting. Next up: party of {size}."
                except Exception:
                    pass

            # 9) Waitlist stats
            waitlist_stats_result = tool_results_by_name.get('getWaitlistStats')
            if waitlist_stats_result:
                try:
                    data = json.loads(waitlist_stats_result)
                    total_waiting = data.get('total_waiting', 0)
                    next_up = data.get('next_up') or {}
                    size = next_up.get('party_size', 'N/A') if isinstance(next_up, dict) else 'N/A'
                    avg = data.get('average_quoted_wait_minutes')
                    avg_text = f", avg quoted {avg} min" if avg is not None else ""
                    return f"Waitlist: {total_waiting} active{avg_text}. Next up: party of {size}."
                except Exception:
                    pass

            # 10) Wait time estimate
            wait_est_result = tool_results_by_name.get('estimateWaitTime')
            if wait_est_result:
                try:
                    data = json.loads(wait_est_result)
                    est = data.get('estimated_wait_minutes')
                    size = data.get('party_size')
                    if est is not None:
                        return f"Estimated wait for party of {size}: ~{est} minutes."
                except Exception:
                    pass

            # 11) Table combination validation
            validation_result = tool_results_by_name.get('validateTableCombination')
            if validation_result:
                try:
                    data = json.loads(validation_result)
                    is_valid = data.get('is_valid', False)
                    capacity = data.get('total_capacity', 0)
                    message = data.get('validation_message', '')
                    party_size = data.get('party_size', 0)
                    valid_text = "✅ Valid" if is_valid else "❌ Invalid"
                    return f"{valid_text} table combination for party of {party_size}. Total capacity: {capacity}. {message}"
                except Exception:
                    pass

            # 12) Availability report
            report_result = tool_results_by_name.get('getTableAvailabilityReport')
            if report_result:
                try:
                    data = json.loads(report_result)
                    date = data.get('date', 'Unknown date')
                    hourly = data.get('hourly_availability', [])
                    summary = data.get('summary', {})
                    peak_hours = summary.get('peak_hours', [])
                    quiet_hours = summary.get('quiet_hours', [])
                    
                    lines = [f"Table availability report for {date}:"]
                    if peak_hours:
                        peak_times = [f"{h.get('hour', '?')}:00" for h in peak_hours[:3]]
                        lines.append(f"Peak hours: {', '.join(peak_times)}")
                    if quiet_hours:
                        quiet_times = [f"{h.get('hour', '?')}:00" for h in quiet_hours[:3]]
                        lines.append(f"Quiet hours: {', '.join(quiet_times)}")
                    
                    return "\n".join(lines).strip()
                except Exception:
                    pass
            
            # Fallback to last AI message content
            if ai_messages:
                last_ai_message = ai_messages[-1]
                final_response = last_ai_message.content or "I apologize, but I couldn't generate a proper response. Please try again."
                
                # Save conversation to memory if provided
                if memory:
                    memory.add_message(user_message)
                    memory.add_message(last_ai_message)
                
                return final_response
            else:
                print("No AI messages found in result")
                return "Sorry, I couldn't process your request."
                
        finally:
            # Always restore the original client
            supabase = original_client
            
    except Exception as e:
        # Ensure we restore the original client even if there's an error
        if 'original_client' in locals():
            supabase = original_client
        print(f"Error running staff agent: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# Interactive chat function for testing
def start_staff_interactive_chat():
    """Start an interactive chat session for local testing."""
    print("🍽️ Welcome to Restaurant Staff AI Assistant!")
    print("I'm here to help restaurant staff work more efficiently.")
    print("Ask me about:")
    print("- Table assignments and availability")
    print("- Customer history and preferences") 
    print("- Today's bookings and statistics")
    print("- Operational insights and suggestions")
    print("\nType 'quit' to exit or enter your question.")
    print("-" * 50)
    
    # For testing, use the provided restaurant ID
    test_restaurant_id = input("Enter restaurant ID (or press Enter for demo): ").strip()
    if not test_restaurant_id:
        test_restaurant_id = "660e8400-e29b-41d4-a716-446655440001"
    
    while True:
        user_input = input("\nStaff: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thanks for using Restaurant Staff AI! Goodbye! 👋")
            break
        elif not user_input:
            print("Please enter a question.")
            continue
        
        print("AI Assistant: ", end="", flush=True)
        response = chat_with_staff_bot(user_input, test_restaurant_id)
        print(response)

# Example usage for testing
if __name__ == "__main__":
    # Only start interactive chat if directly run, not when imported
    start_staff_interactive_chat()

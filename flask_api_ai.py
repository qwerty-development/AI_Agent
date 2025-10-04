from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
import jwt
from supabase import create_client, Client
from functools import wraps
import hashlib
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple in-memory response cache to prevent duplicate LLM calls
# Cache structure: {cache_key: (response, timestamp)}
_RESPONSE_CACHE = {}
_CACHE_TTL = timedelta(minutes=5)  # Cache responses for 5 minutes
_MAX_CACHE_SIZE = 100  # Prevent memory bloat

def _get_cache_key(message: str, user_id: str = None) -> str:
    """Generate cache key from message and user_id"""
    # Normalize message (lowercase, strip whitespace)
    normalized = message.lower().strip()
    # Include user_id for personalized responses
    cache_str = f"{normalized}:{user_id or 'anon'}"
    return hashlib.md5(cache_str.encode()).hexdigest()

def _get_cached_response(cache_key: str) -> dict:
    """Get cached response if it exists and is still valid"""
    if cache_key in _RESPONSE_CACHE:
        response, timestamp = _RESPONSE_CACHE[cache_key]
        if datetime.now() - timestamp < _CACHE_TTL:
            logger.info(f"Cache HIT for key: {cache_key[:8]}...")
            return response
        else:
            # Expired, remove from cache
            del _RESPONSE_CACHE[cache_key]
            logger.info(f"Cache EXPIRED for key: {cache_key[:8]}...")
    return None

def _set_cached_response(cache_key: str, response: dict):
    """Cache a response with current timestamp"""
    # Prevent cache from growing too large
    if len(_RESPONSE_CACHE) >= _MAX_CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_RESPONSE_CACHE))
        del _RESPONSE_CACHE[oldest_key]
        logger.info(f"Cache FULL - removed oldest entry")

    _RESPONSE_CACHE[cache_key] = (response, datetime.now())
    logger.info(f"Cache SET for key: {cache_key[:8]}...")

app = Flask(__name__)
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Request logging middleware
@app.before_request
def log_request_info():
    """Log basic request information for monitoring"""
    logger.info(f'Request: {request.method} {request.path} from {request.remote_addr} - User-Agent: {request.headers.get("User-Agent", "Unknown")}')

# Security headers middleware
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'none'; object-src 'none';"
    return response

# Authentication helper functions
def create_authenticated_supabase_client(jwt_token):
    """Create a Supabase client with user authentication"""
    try:
        url = os.getenv("EXPO_PUBLIC_SUPABASE_URL")
        key = os.getenv("EXPO_PUBLIC_SUPABASE_ANON_KEY")
        
        if not url or not key:
            logger.error("Missing Supabase environment variables")
            return None
            
        # Create client with user JWT token
        client = create_client(url, key)
        
        # Set JWT token in headers for authenticated requests
        client.postgrest.auth(jwt_token)
        
        return client
    except Exception as e:
        logger.error(f"Failed to create authenticated Supabase client: {e}")
        return None

def extract_jwt_token():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    return None

def get_user_from_token(jwt_token):
    """Extract user info from JWT token (without verification for now)"""
    try:
        # For Supabase, we'll let the client handle token validation
        # The token format contains user info we can extract
        import base64
        import json
        
        # Split JWT to get payload
        parts = jwt_token.split('.')
        if len(parts) != 3:
            return None
            
        # Decode payload (add padding if needed)
        payload = parts[1]
        payload += '=' * (4 - len(payload) % 4)  # Add padding
        decoded = base64.b64decode(payload)
        user_data = json.loads(decoded)
        
        return {
            'id': user_data.get('sub'),
            'email': user_data.get('email'),
            'role': user_data.get('role', 'authenticated')
        }
    except Exception as e:
        logger.warning(f"Failed to extract user from token: {e}")
        return None

# Simple request validation
def validate_request():
    """Simple validation to ensure requests come from expected sources"""
    user_agent = request.headers.get('User-Agent', '')
    referer = request.headers.get('Referer', '')
    
    # Allow requests from known frontends or mobile apps
    allowed_patterns = [
        'expo',  # Expo/React Native apps
        'okhttp',  # Android apps
        'CFNetwork',  # iOS apps
        'chrome',  # Web browsers
        'firefox',
        'safari',
        'edge'
    ]
    
    # Check if request comes from a reasonable source
    if any(pattern.lower() in user_agent.lower() for pattern in allowed_patterns):
        return True
    
    # Allow requests with no user agent (for development/testing)
    if not user_agent:
        return True
        
    return False

def require_valid_request(f):
    """Decorator to validate requests"""
    def decorated_function(*args, **kwargs):
        if not validate_request():
            return jsonify({
                'error': 'Invalid request source',
                'status': 'error'
            }), 403
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Rate limit error handler
@app.errorhandler(429)
def ratelimit_handler(e):
    response = jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please wait before making another request.',
        'status': 'error'
    })
    
    # Add rate limit headers for frontend handling
    response.headers['Retry-After'] = '60'  # Suggest retry after 60 seconds
    response.headers['X-RateLimit-Limit'] = getattr(e.description, 'limit', 'unknown')
    response.headers['X-RateLimit-Remaining'] = '0'
    
    return response, 429

# Try to import AI functionality with graceful fallback
AI_AVAILABLE = False
STAFF_AI_AVAILABLE = False

try:
    from AI_Agent import chat_with_bot, getAllCuisineTypes, create_conversation_memory
    AI_AVAILABLE = True
    logger.info("AI Agent imported successfully")
except Exception as e:
    logger.warning(f"AI Agent import failed: {e}")
    AI_AVAILABLE = False

try:
    from AI_Agent_Restaurant import chat_with_staff_bot, create_staff_conversation_memory
    STAFF_AI_AVAILABLE = True
    logger.info("Restaurant Staff AI Agent imported successfully")
except Exception as e:
    logger.warning(f"Restaurant Staff AI Agent import failed: {e}")
    STAFF_AI_AVAILABLE = False

@app.route('/', methods=['GET'])
@limiter.exempt  # Exempt home page from rate limiting
def home():
    return jsonify({
        'status': 'healthy',
        'message': 'Restaurant AI Agent API is running - v2.0',
        'ai_available': AI_AVAILABLE,
        'staff_ai_available': STAFF_AI_AVAILABLE,
        'environment_check': {
            'GOOGLE_API_KEY': 'Set' if os.getenv('GOOGLE_API_KEY') else 'Missing',
            'EXPO_PUBLIC_SUPABASE_URL': 'Set' if os.getenv('EXPO_PUBLIC_SUPABASE_URL') else 'Missing',
            'EXPO_PUBLIC_SUPABASE_ANON_KEY': 'Set' if os.getenv('EXPO_PUBLIC_SUPABASE_ANON_KEY') else 'Missing'
        }
    }), 200

@app.route('/api/health', methods=['GET'])
@limiter.exempt  # Exempt health checks from rate limiting
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Restaurant AI Agent API is running',
        'ai_available': AI_AVAILABLE,
        'staff_ai_available': STAFF_AI_AVAILABLE
    }), 200

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")  # Allow 30 chat requests per minute per IP
@require_valid_request
def chat():
    """
    Main chat endpoint to send messages to the AI agent.
    Supports conversation history for contextual responses.
    Now supports authenticated users with RLS.
    """
    try:
        if not AI_AVAILABLE:
            return jsonify({
                'error': 'AI functionality not available',
                'message': 'Please check environment variables and dependencies',
                'status': 'error'
            }), 503

        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required',
                'status': 'error'
            }), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id', 'default')
        user_id = data.get('user_id')  # Optional user ID for personalization
        conversation_history = data.get('conversation_history', [])  # New parameter
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty',
                'status': 'error'
            }), 400
        
        # Extract JWT token and create authenticated client
        jwt_token = extract_jwt_token()
        authenticated_client = None
        current_user = None
        
        if jwt_token:
            authenticated_client = create_authenticated_supabase_client(jwt_token)
            current_user = get_user_from_token(jwt_token)
            if current_user:
                logger.info(f"Authenticated request from user {current_user['id']} (email: {current_user.get('email', 'unknown')})")
            else:
                logger.warning("JWT token present but user extraction failed")
        else:
            logger.info("No JWT token provided, using anonymous access")
        
        logger.info(f"Received message from session {session_id} (user: {user_id}): {user_message}")
        logger.info(f"Conversation history length: {len(conversation_history) if conversation_history else 0}")

        # Check cache for identical recent queries (only for non-conversational requests)
        # Skip cache if conversation history exists (context-dependent responses)
        cache_key = None
        if not conversation_history or len(conversation_history) == 0:
            cache_key = _get_cache_key(user_message, user_id)
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Returning cached response for: {user_message[:50]}...")
                cached_response['cached'] = True  # Add cache indicator
                return jsonify(cached_response), 200

        # Convert conversation history to LangChain message format if provided
        memory = None
        if conversation_history and isinstance(conversation_history, list):
            try:
                from AI_Agent import create_conversation_memory
                from langchain_core.messages import HumanMessage, AIMessage

                memory = create_conversation_memory(max_history=6)
                
                # Convert frontend message format to LangChain messages
                for msg in conversation_history:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        content = msg['content'].strip()
                        if content:  # Only add non-empty messages
                            if msg['role'] == 'user':
                                memory.add_message(HumanMessage(content=content))
                            elif msg['role'] == 'assistant':
                                memory.add_message(AIMessage(content=content))
                        
                logger.info(f"Converted {memory.get_context_size()} messages to conversation memory")
            except Exception as e:
                logger.warning(f"Failed to process conversation history: {e}")
                memory = None
        
        # Get AI response with conversation context and authentication
        ai_response = chat_with_bot(
            user_message, 
            memory=memory, 
            user_id=user_id,
            authenticated_client=authenticated_client,
            current_user=current_user
        )
        
        logger.info(f"AI response for session {session_id}: {ai_response}")
        
        # Check if response contains restaurant recommendations
        restaurants_to_show = []
        response_text = ai_response
        
        if "RESTAURANTS_TO_SHOW:" in ai_response:
            parts = ai_response.split("RESTAURANTS_TO_SHOW:")
            response_text = parts[0].strip()
            if len(parts) > 1:
                restaurant_ids = [id.strip() for id in parts[1].strip().split(',') if id.strip()]
                restaurants_to_show = restaurant_ids
        
        response_data = {
            'response': response_text,
            'restaurants_to_show': restaurants_to_show,
            'session_id': session_id,
            'user_id': user_id,
            'authenticated': jwt_token is not None,
            'cached': False,
            'status': 'success'
        }

        # Cache the response for future identical queries (only for non-conversational)
        if cache_key:
            _set_cached_response(cache_key, response_data)

        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/chat/reset', methods=['POST'])
def chat_reset():
    """
    Reset conversation endpoint.
    Provides acknowledgment for conversation reset requests.
    Note: Actual conversation history is managed by the frontend.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default') if data else 'default'
        user_id = data.get('user_id') if data else None
        clear_history = data.get('clear_history', True) if data else True
        
        logger.info(f"Conversation reset requested for session {session_id} (user: {user_id}), clear_history: {clear_history}")
        
        return jsonify({
            'message': 'Conversation reset successfully',
            'session_id': session_id,
            'user_id': user_id,
            'clear_history': clear_history,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat reset endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/restaurants/cuisines', methods=['GET'])
@limiter.limit("10 per minute")  # Lower limit for cuisine endpoint
def get_cuisine_types():
    """Get all available cuisine types."""
    try:
        if not AI_AVAILABLE:
            return jsonify({
                'error': 'AI functionality not available',
                'status': 'error'
            }), 503

        # Call the tool safely to get cuisine types (@tool wraps it as a Tool object)
        try:
            # Preferred path for no-arg tools
            cuisine_types_str = getAllCuisineTypes.invoke({})
        except Exception:
            try:
                # Some versions accept a positional empty string
                cuisine_types_str = getAllCuisineTypes("")
            except Exception:
                try:
                    # Older API
                    cuisine_types_str = getAllCuisineTypes.run("")
                except Exception:
                    try:
                        # Access original function if exposed
                        func = getattr(getAllCuisineTypes, "func", None)
                        cuisine_types_str = func() if callable(func) else "[]"
                    except Exception:
                        cuisine_types_str = "[]"
        
        # Parse the JSON response
        import json
        try:
            cuisine_types = json.loads(cuisine_types_str)
        except:
            cuisine_types = []
        
        return jsonify({
            'cuisine_types': cuisine_types,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cuisine types: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/staff/chat', methods=['POST'])
@limiter.limit("50 per minute")  # Allow more requests for staff
@require_valid_request
def staff_chat():
    """
    Chat endpoint specifically for restaurant staff assistance.
    Requires restaurant_id in the request.
    Supports conversation history for contextual responses.
    Now supports authenticated staff with RLS.
    """
    try:
        if not STAFF_AI_AVAILABLE:
            return jsonify({
                'error': 'Staff AI functionality not available',
                'message': 'Please check environment variables and dependencies',
                'status': 'error'
            }), 503

        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required',
                'status': 'error'
            }), 400
        
        user_message = data['message'].strip()
        restaurant_id = data.get('restaurant_id')
        session_id = data.get('session_id', 'staff_default')
        user_id = data.get('user_id')  # Staff user ID
        conversation_history = data.get('conversation_history', [])  # New parameter
        
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty',
                'status': 'error'
            }), 400
        
        # Extract JWT token and create authenticated client
        jwt_token = extract_jwt_token()
        authenticated_client = None
        current_user = None
        
        if jwt_token:
            authenticated_client = create_authenticated_supabase_client(jwt_token)
            current_user = get_user_from_token(jwt_token)
            if current_user:
                logger.info(f"Authenticated staff request from user {current_user['id']} (email: {current_user.get('email', 'unknown')})")
            else:
                logger.warning("JWT token present but user extraction failed")
        else:
            logger.info("No JWT token provided for staff request")
        
        logger.info(f"Received staff message from session {session_id}: {user_message}")
        logger.info(f"Staff conversation history length: {len(conversation_history) if conversation_history else 0}")
        
        # Convert conversation history to LangChain message format if provided
        memory = None
        if conversation_history and isinstance(conversation_history, list):
            try:
                from langchain_core.messages import HumanMessage, AIMessage

                memory = create_staff_conversation_memory(max_history=6)
                
                # Convert frontend message format to LangChain messages
                for msg in conversation_history:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        content = msg['content'].strip()
                        if content:  # Only add non-empty messages
                            if msg['role'] == 'user':
                                memory.add_message(HumanMessage(content=content))
                            elif msg['role'] == 'assistant':
                                memory.add_message(AIMessage(content=content))
                        
                logger.info(f"Converted {memory.get_context_size()} messages to staff conversation memory")
            except Exception as e:
                logger.warning(f"Failed to process staff conversation history: {e}")
                memory = None
        
        # Get Staff AI response with conversation context and authentication
        staff_response = chat_with_staff_bot(
            user_message, 
            restaurant_id, 
            memory=memory,
            authenticated_client=authenticated_client,
            current_user=current_user
        )
        
        logger.info(f"Staff AI response for session {session_id}: {staff_response}")
        
        return jsonify({
            'response': staff_response,
            'session_id': session_id,
            'restaurant_id': restaurant_id,
            'user_id': user_id,
            'authenticated': jwt_token is not None,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in staff chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/test', methods=['POST'])
@require_valid_request
def test_endpoint():
    """Test endpoint for debugging purposes."""
    try:
        data = request.get_json()
        
        return jsonify({
            'message': 'Test endpoint working',
            'received_data': data,
            'ai_available': AI_AVAILABLE,
            'staff_ai_available': STAFF_AI_AVAILABLE,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/admin/stats', methods=['GET'])
@limiter.limit("5 per minute")
def admin_stats():
    """Simple admin endpoint to get basic stats - protect this in production"""
    admin_key = request.headers.get('X-Admin-Key')
    expected_key = os.getenv('ADMIN_KEY', 'admin123')  # Change this in production
    
    if admin_key != expected_key:
        return jsonify({
            'error': 'Unauthorized',
            'status': 'error'
        }), 401
    
    try:
        # Get current time for stats
        from datetime import datetime
        current_time = datetime.utcnow().isoformat()
        
        # Return enhanced stats
        return jsonify({
            'status': 'healthy',
            'ai_available': AI_AVAILABLE,
            'staff_ai_available': STAFF_AI_AVAILABLE,
            'timestamp': current_time,
            'message': 'Admin stats endpoint',
            'rate_limits': {
                'default': "200 per day, 50 per hour",
                'chat': "30 per minute",
                'staff_chat': "50 per minute",
                'cuisines': "10 per minute",
                'admin': "5 per minute"
            },
            'security_features': {
                'request_validation': True,
                'security_headers': True,
                'request_logging': True,
                'admin_protection': True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in admin stats endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
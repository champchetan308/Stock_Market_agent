# common/gemini_utils.py
import os
import logging
import time
import json
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, RetryError, Aborted, DeadlineExceeded

logger = logging.getLogger(__name__)

# --- Environment Variable for API Key ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Global Model Configuration ---
DEFAULT_GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" # As per user request for latest Pro model

# Default Generation Configuration - expecting JSON output
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.5, # Lower for more deterministic, structured output for trading
    "top_p": 0.9,
    "top_k": 30,
    "max_output_tokens": 4096, # Increased for potentially complex JSONs / reasoning
    "response_mime_type": "application/json", 
}

DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, # More permissive for financial analysis
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}, # Block clearly dangerous
]

_initialized_models_cache = {} # Cache for model instances

def get_gemini_model_instance(model_name=DEFAULT_GEMINI_MODEL_NAME, 
                              generation_config_override=None, 
                              safety_settings_override=None):
    """
    Configures and returns a Gemini GenerativeModel instance. Caches initialized models.
    """
    global _initialized_models_cache

    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY environment variable not set. Cannot use Gemini models.")
        raise ValueError("GEMINI_API_KEY not set.")

    gen_config = generation_config_override if generation_config_override is not None else DEFAULT_GENERATION_CONFIG
    safety_config = safety_settings_override if safety_settings_override is not None else DEFAULT_SAFETY_SETTINGS
    
    model_config_key = f"{model_name}_{json.dumps(gen_config, sort_keys=True)}_{json.dumps(safety_config, sort_keys=True)}"

    if model_config_key in _initialized_models_cache:
        return _initialized_models_cache[model_config_key]

    try:
        # Ensure genai is configured with the API key
        # This might be called multiple times but is idempotent for the key.
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            safety_settings=safety_config
        )
        _initialized_models_cache[model_config_key] = model_instance
        logger.info(f"Gemini model '{model_name}' initialized and cached with specified config.")
        return model_instance
    except Exception as e:
        logger.error(f"Error configuring Gemini model '{model_name}': {e}", exc_info=True)
        raise # Re-raise to indicate failure

def generate_structured_gemini_response(prompt_text, 
                                        model_name=DEFAULT_GEMINI_MODEL_NAME,
                                        retry_attempts=3, 
                                        initial_backoff_sec=3,
                                        request_timeout_sec=120): # Timeout for the API call
    """
    Generates a structured JSON response from Gemini, with robust error handling and retries.
    """
    if not prompt_text or not isinstance(prompt_text, str):
        logger.error("Invalid or empty prompt_text provided to Gemini.")
        return {"error": "INVALID_PROMPT", "message": "Prompt text is empty or invalid."}

    try:
        model = get_gemini_model_instance(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to get Gemini model instance '{model_name}': {e}")
        return {"error": "MODEL_INIT_FAILURE", "message": str(e)}

    for attempt in range(retry_attempts):
        try:
            logger.info(f"Attempt {attempt + 1}/{retry_attempts} to generate Gemini response (model: '{model_name}').")
            logger.debug(f"Prompt (first 500 chars):\n{prompt_text[:500]}...")
            
            # For models supporting request_options (like timeout)
            # request_options = {"timeout": request_timeout_sec} if request_timeout_sec else {}
            # response = model.generate_content(prompt_text, request_options=request_options)
            # Simpler call if timeout is handled by SDK's default or not explicitly needed here:
            response = model.generate_content(prompt_text)


            if not response.candidates:
                error_msg = "Gemini response has no candidates."
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    error_msg += f" Feedback: {response.prompt_feedback}"
                    if response.prompt_feedback.block_reason:
                        logger.error(f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason_message}")
                        return {"error": "PROMPT_BLOCKED", "message": response.prompt_feedback.block_reason_message}
                logger.warning(error_msg)
                # Retry for "no candidates" if not explicitly blocked, as it might be transient.
                if attempt < retry_attempts - 1:
                    time.sleep(initial_backoff_sec * (2 ** attempt))
                    continue
                return {"error": "NO_CANDIDATES", "message": error_msg}

            candidate = response.candidates[0]
            # Finish reason: 1 ("STOP"), 2 ("MAX_TOKENS"), 3 ("SAFETY"), 4 ("RECITATION"), 5 ("OTHER")
            if candidate.finish_reason != genai.types. όμωςFinishReason.STOP:
                error_message = f"Gemini generation finished unsuccessfully. Reason: {candidate.finish_reason.name} ({candidate.finish_reason.value})."
                if hasattr(candidate, 'finish_message') and candidate.finish_message:
                    error_message += f" Message: {candidate.finish_message}"
                # Log safety ratings if present and finish reason is SAFETY
                if candidate.finish_reason == genai.types. όμωςFinishReason.SAFETY and candidate.safety_ratings:
                    error_message += f" Safety Ratings: {[str(sr) for sr in candidate.safety_ratings]}"
                logger.error(error_message)
                # Do not retry for definitive non-successful completions like SAFETY or RECITATION on final attempt
                if candidate.finish_reason in [genai.types. όμωςFinishReason.SAFETY, genai.types. όμωςFinishReason.RECITATION] or attempt == retry_attempts - 1:
                    return {"error": f"FINISH_REASON_{candidate.finish_reason.name}", "message": error_message}
                # Retry for MAX_TOKENS or OTHER if attempts remain
                time.sleep(initial_backoff_sec * (2 ** attempt))
                continue


            if not candidate.content or not candidate.content.parts:
                logger.error("Gemini response candidate has no content parts.")
                return {"error": "NO_CONTENT_PARTS", "message": "AI response content was empty."}

            try:
                # response_mime_type="application/json" should ensure text is JSON string.
                json_text = candidate.content.parts[0].text
                parsed_json = json.loads(json_text)
                logger.info(f"Successfully parsed JSON response from Gemini model '{model_name}'.")
                return parsed_json
            except (json.JSONDecodeError, IndexError, AttributeError) as json_e:
                raw_text_snippet = candidate.content.parts[0].text[:500] if candidate.content.parts else "N/A"
                logger.error(f"Failed to parse JSON from Gemini. Error: {json_e}. Raw text snippet: '{raw_text_snippet}'", exc_info=True)
                # Don't retry JSON parse errors, it's a content issue.
                return {"error": "JSON_PARSE_ERROR", "message": str(json_e), "raw_text": raw_text_snippet}
            
        except (RetryError, Aborted, DeadlineExceeded) as e: # Specific retryable or timeout errors
            logger.warning(f"Google API retryable/timeout error on attempt {attempt + 1}: {type(e).__name__} - {e}. Retrying...")
        except GoogleAPIError as e: # Other Google API errors
            logger.error(f"Google API error on attempt {attempt + 1}: {e}. Retrying if applicable...", exc_info=True)
            # Check if error is fatal (e.g., auth, quota permanently exhausted)
            if e.grpc_status_code == 7: #PERMISSION_DENIED (e.g. API not enabled, bad key)
                 logger.critical(f"Fatal GoogleAPIError (Permission Denied): {e}. Aborting retries.")
                 return {"error": "API_PERMISSION_DENIED", "message": str(e)}
            # Add other fatal error codes here
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during Gemini call on attempt {attempt + 1}: {e}", exc_info=True)
        
        if attempt < retry_attempts - 1:
            wait_time = initial_backoff_sec * (2 ** attempt)
            logger.info(f"Waiting {wait_time} seconds before next Gemini attempt...")
            time.sleep(wait_time)
            
    logger.error(f"All {retry_attempts} attempts to call Gemini model '{model_name}' failed.")
    return {"error": "MAX_RETRIES_EXCEEDED", "message": f"Failed after {retry_attempts} attempts."}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("Please set the GEMINI_API_KEY environment variable to test gemini_utils.py")
    else:
        logger.info("--- Testing Gemini Structured JSON Response (gemini-1.5-pro-latest) ---")
        test_prompt = """
        Provide a JSON object with keys "instrument_name", "action", and "confidence_level".
        Instrument name should be "NIFTYBANK", action should be "BUY", confidence level should be 0.78.
        """
        json_response = generate_structured_gemini_response(test_prompt, model_name="gemini-1.5-pro-latest") # Explicitly test Pro
        
        if json_response and "error" not in json_response:
            logger.info(f"\nResponse:\n{json.dumps(json_response, indent=2)}")
            assert json_response.get("instrument_name") == "NIFTYBANK"
            assert json_response.get("action") == "BUY"
        elif json_response: # Error structure returned
            logger.error(f"Gemini call resulted in an error structure: {json_response}")
        else: # None returned (should be covered by error structure now)
            logger.error("Failed to get any response for the JSON test prompt.")

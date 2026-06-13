import json
import logging
from json import JSONDecodeError

logger = logging.getLogger(__name__)


def parse_llm_json(raw_output: str) -> dict | list:
    """
    Robustly parse JSON from an LLM response.
    Handles Markdown code fences, trailing text, and nested structures
    by finding the first JSON-like structure and decoding it.
    """
    if isinstance(raw_output, list):
        # Merge parts if it's a list of blocks/parts
        parts = []
        for part in raw_output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        raw_output = "".join(parts)
    elif not isinstance(raw_output, str):
        raw_output = str(raw_output)

    cleaned = raw_output.strip()
    
    # Fast path: if it looks like clean JSON, try parsing immediately
    if (cleaned.startswith("{") and cleaned.endswith("}")) or \
       (cleaned.startswith("[") and cleaned.endswith("]")):
        try:
            return json.loads(cleaned)
        except JSONDecodeError:
            pass
            
    # Remove markdown code fences if present at the boundaries
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove the first line (e.g. ```json)
        if len(lines) > 1:
            lines = lines[1:]
        # Remove the last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except JSONDecodeError:
            pass

    # Fallback: scan for the first '{' or '[' and try to decode using raw_decode
    start_idx_dict = cleaned.find('{')
    start_idx_list = cleaned.find('[')
    
    start_idx = -1
    if start_idx_dict != -1 and start_idx_list != -1:
        start_idx = min(start_idx_dict, start_idx_list)
    elif start_idx_dict != -1:
        start_idx = start_idx_dict
    elif start_idx_list != -1:
        start_idx = start_idx_list
        
    if start_idx != -1:
        try:
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(cleaned[start_idx:])
            return result
        except JSONDecodeError as e:
            logger.debug(f"JSON raw_decode failed at index {start_idx}: {e}")
            logger.debug(f"Raw content: {raw_output}")
    
    # If all else fails, raise a helpful error
    raise ValueError(f"Could not extract valid JSON from LLM output. Raw: {raw_output[:100]}...")


def parse_llm_json_list(raw_output: str) -> list:
    """
    Wrapper for parse_llm_json that ensures the result is a list.
    """
    result = parse_llm_json(raw_output)
    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON list, but got {type(result).__name__}")
    return result

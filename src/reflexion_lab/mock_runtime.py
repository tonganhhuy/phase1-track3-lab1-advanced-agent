from __future__ import annotations
import os
import json
import time
import urllib.request
import urllib.error
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

load_dotenv()

# Global config
LIVE_MODE = os.environ.get("LIVE_MODE", "false").lower() in ("true", "1", "yes")

# Metrics tracker for actual LLM calls
class MetricsTracker:
    def __init__(self):
        self.step_tokens = 0
        self.step_latency = 0
        
    def reset_step(self):
        self.step_tokens = 0
        self.step_latency = 0
        
    def add_call(self, tokens: int, latency: int):
        self.step_tokens += tokens
        self.step_latency += latency

metrics_tracker = MetricsTracker()

# Track last request timestamp to enforce rate limit (RPM)
_last_request_time = 0.0

def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def call_gemini_api(system_prompt: str, user_prompt: str, response_json: bool = False, model: str = "gemini-3.1-flash-lite") -> tuple[str, int, int]:
    """
    Calls the Gemini API using urllib with strict rate-limiting (RPM) and backoff retry.
    Returns: (response_text, tokens, latency_ms)
    """
    global _last_request_time
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file or environment.")
        
    # Read model and rate limit configuration
    model = os.environ.get("GEMINI_MODEL", model)
    # Default to 15 RPM to fit Google AI Studio free tier limits safely
    rpm = float(os.environ.get("GEMINI_RPM", "15"))
    min_interval = 60.0 / rpm if rpm > 0 else 0.0
    
    # Enforce rate limit (minimum time spacing between requests)
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        time.sleep(sleep_time)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": user_prompt}
            ]
        }
    ]
    
    payload = {
        "contents": contents
    }
    
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [
                {"text": system_prompt}
            ]
        }
        
    if response_json:
        payload["generationConfig"] = {
            "responseMimeType": "application/json"
        }
        
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    
    max_retries = 10
    backoff = 4.0
    
    for attempt in range(max_retries):
        start_time = time.time()
        _last_request_time = start_time
        
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as response:
                latency_ms = int((time.time() - start_time) * 1000)
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                
                candidates = res_json.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Empty candidates in Gemini response: {res_body}")
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"No parts in Gemini candidate: {res_body}")
                
                text_out = parts[0].get("text", "")
                
                # Extract token usage
                usage = res_json.get("usageMetadata", {})
                tokens = usage.get("totalTokenCount", 0)
                if tokens == 0:
                    tokens = len(user_prompt.split()) + len(text_out.split()) + 100
                
                return text_out, tokens, latency_ms
                
        except urllib.error.HTTPError as e:
            status_code = e.code
            err_body = e.read().decode("utf-8", errors="ignore")
            
            if status_code in (429, 503, 500) and attempt < max_retries - 1:
                sleep_sec = backoff * (2.0 ** attempt)
                # Parse retryDelay from Google RPC details if available
                try:
                    err_json = json.loads(err_body)
                    details = err_json.get("error", {}).get("details", [])
                    for detail in details:
                        if "RetryInfo" in detail.get("@type", ""):
                            delay_str = detail.get("retryDelay", "")
                            if delay_str.endswith("s"):
                                # Extract numeric value (e.g. 30.906050598s -> 30.9)
                                sleep_sec = float(delay_str[:-1]) + 2.0
                                print(f"[RateLimit/Error] Parsed Google RPC retryDelay: {sleep_sec:.2f} seconds.")
                                break
                except Exception:
                    pass
                print(f"[RateLimit/Error] HTTP {status_code} received. Retrying in {sleep_sec:.2f} seconds...")
                time.sleep(sleep_sec)
                continue
            else:
                print(f"[API Error] Failed with HTTP {status_code}: {err_body}")
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_sec = backoff * (2.0 ** attempt)
                print(f"[Connection Error] {str(e)}. Retrying in {sleep_sec:.2f} seconds...")
                time.sleep(sleep_sec)
                continue
            else:
                raise e
                
    raise RuntimeError("Failed to call Gemini API after max retries")

# Original Mock data & mock behavior
FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if not LIVE_MODE:
        if example.qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[example.qid]
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[example.qid]
        return example.gold_answer

    # Live Mode implementation
    context_str = "\n\n".join([f"Document: {chunk.title}\n{chunk.text}" for chunk in example.context])
    user_prompt = f"Question: {example.question}\n\nContext:\n{context_str}"
    
    if reflection_memory:
        reflections_str = "\n".join([f"- Attempt {i+1}: {ref}" for i, ref in enumerate(reflection_memory)])
        user_prompt += f"\n\nPast Reflections/Lessons from failed attempts:\n{reflections_str}\nAvoid repeating these mistakes!"
        
    answer, tokens, latency = call_gemini_api(ACTOR_SYSTEM, user_prompt, response_json=False)
    metrics_tracker.add_call(tokens, latency)
    return answer.strip()

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if not LIVE_MODE:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

    # Live Mode implementation
    user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"
    eval_text, tokens, latency = call_gemini_api(EVALUATOR_SYSTEM, user_prompt, response_json=True)
    metrics_tracker.add_call(tokens, latency)
    
    cleaned_text = clean_json_text(eval_text)
    try:
        data = json.loads(cleaned_text)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "No reason provided.")),
            missing_evidence=data.get("missing_evidence"),
            spurious_claims=data.get("spurious_claims")
        )
    except Exception:
        # Fallback evaluation if JSON parse fails
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        return JudgeResult(
            score=1 if is_correct else 0,
            reason=f"Parsed failed. Semantic equivalence: {is_correct}. Raw feedback: {cleaned_text[:150]}",
            missing_evidence=None,
            spurious_claims=[answer] if not is_correct else None
        )

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if not LIVE_MODE:
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

    # Live Mode implementation
    wrong_answer = "Unknown wrong answer"
    if judge.spurious_claims and len(judge.spurious_claims) > 0:
        wrong_answer = judge.spurious_claims[0]
        
    user_prompt = f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Incorrect Predicted Answer: {wrong_answer}
Evaluator Reason: {judge.reason}
Missing Evidence: {judge.missing_evidence}
Spurious Claims: {judge.spurious_claims}"""

    ref_text, tokens, latency = call_gemini_api(REFLECTOR_SYSTEM, user_prompt, response_json=True)
    metrics_tracker.add_call(tokens, latency)
    
    cleaned_text = clean_json_text(ref_text)
    try:
        data = json.loads(cleaned_text)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "Verify entities carefully.")),
            next_strategy=str(data.get("next_strategy", "Search context for missing links."))
        )
    except Exception:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection JSON.",
            next_strategy="Identify the missing evidence and trace the correct connection from the context."
        )

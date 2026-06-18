from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        from .mock_runtime import metrics_tracker, LIVE_MODE, FAILURE_MODE_BY_QID

        # Extension: adaptive_max_attempts
        current_max_attempts = self.max_attempts
        if self.agent_type == "reflexion":
            if example.difficulty == "easy":
                current_max_attempts = 1
            elif example.difficulty == "medium":
                current_max_attempts = 3
            elif example.difficulty == "hard":
                current_max_attempts = 4

        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_score = 0
        final_answer = ""
        
        for attempt_id in range(1, current_max_attempts + 1):
            metrics_tracker.reset_step()
            
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            
            reflection = None
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < current_max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(reflection.next_strategy)

            # Retrieve actual tokens and latency if in live mode, else use mock estimates
            if LIVE_MODE:
                token_estimate = metrics_tracker.step_tokens
                latency_ms = metrics_tracker.step_latency
            else:
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
            
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection,
                token_estimate=token_estimate,
                latency_ms=latency_ms
            )
            traces.append(trace)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                break
                
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        
        # Determine failure mode: map to known failure modes if failed in live mode
        if final_score == 1:
            failure_mode = "none"
        else:
            if LIVE_MODE:
                # Map to standard failure modes based on evaluator reasons
                reason_lower = traces[-1].reason.lower() if traces else ""
                if "evidence" in reason_lower or "missing" in reason_lower:
                    failure_mode = "incomplete_multi_hop"
                elif "birthplace" in reason_lower:
                    failure_mode = "incomplete_multi_hop"
                elif "wrong" in reason_lower or "selected the wrong" in reason_lower:
                    failure_mode = "wrong_final_answer"
                elif "looping" in reason_lower or "repeat" in reason_lower:
                    failure_mode = "looping"
                elif "reflection" in reason_lower or "overfit" in reason_lower:
                    failure_mode = "reflection_overfit"
                else:
                    failure_mode = "wrong_final_answer"
            else:
                failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
                
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)

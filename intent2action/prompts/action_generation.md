Infer structured action candidates from the input. The system must not execute actions.

Return a single JSON object with this shape:
{
  "input_summary": "short summary",
  "input_type": "text or image",
  "extracted_entities": [
    {"name": "name", "value": "value", "entity_type": "type", "confidence": 0.0}
  ],
  "detected_intents": [
    {"intent": "intent_name", "confidence": 0.0, "rationale": "why"}
  ],
  "possible_actions": [
    {
      "action_name": "human readable action",
      "action_type": "ontology style type",
      "description": "what could be done",
      "rationale": "why this action is reasonable",
      "required_inputs": ["input names required"],
      "available_inputs": ["input names already available"],
      "missing_inputs": ["input names still needed"],
      "suggested_tools": ["tool categories or systems"],
      "confidence": 0.0,
      "risk_level": "low|medium|high",
      "execution_mode": "auto_possible|human_approval_required|draft_only|not_recommended",
      "ranking_score": 0.0
    }
  ],
  "clarifying_questions": ["question if needed"],
  "raw_model_output": null,
  "warnings": []
}

Requirements:
- Identify all reasonable actions, not only one action.
- Include available and missing information for each action.
- Suggested tools must be suggestions only.
- Use low risk for summarization, extraction, classification, and drafting internal notes.
- Use medium risk for creating tickets/tasks, drafting emails, notifications, and scheduling.
- Use high risk for sending external emails, deleting data, payments, production changes, deployments, official records, or sensitive data access.
- Use draft_only or human_approval_required for communications.
- Never state that an action has been executed.
- Return only valid JSON.


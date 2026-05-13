"""Streamlit UI for local action inference."""

from typing import Any

import streamlit as st

from intent2action.app.config import get_settings
from intent2action.core.pipeline import ActionInferencePipeline
from intent2action.providers.openai_compatible_client import OpenAICompatibleClientError

settings = get_settings()


@st.cache_resource
def get_pipeline() -> ActionInferencePipeline:
    """Cache the local pipeline for the Streamlit session."""

    return ActionInferencePipeline()


def context_fields() -> dict[str, Any]:
    """Render context inputs."""

    domain = st.text_input("Domain")
    user_role = st.text_input("User role")
    additional_context = st.text_area("Additional context", height=100)
    return {
        key: value
        for key, value in {
            "domain": domain,
            "user_role": user_role,
            "additional_context": additional_context,
        }.items()
        if value
    }


def render_response(response: Any) -> None:
    """Render an inference response."""

    st.subheader("Input summary")
    st.write(response.input_summary)

    st.subheader("Detected intents")
    for intent in response.detected_intents:
        st.progress(intent.confidence, text=f"{intent.intent} ({intent.confidence:.0%})")
        st.caption(intent.rationale)

    st.subheader("Extracted entities")
    if response.extracted_entities:
        st.dataframe(
            [entity.model_dump() for entity in response.extracted_entities],
            hide_index=True,
        )
    else:
        st.caption("No entities detected.")

    st.subheader("Action candidates")
    for action in response.possible_actions:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**{action.action_name}**")
            col2.metric("Confidence", f"{action.confidence:.0%}")
            col3.metric("Rank", f"{action.ranking_score:.2f}")
            st.write(action.description)
            st.caption(action.rationale)
            st.markdown(f"Risk: `{action.risk_level.value}`")
            st.write(f"Execution mode: `{action.execution_mode.value}`")
            st.write("Required inputs:", action.required_inputs or "None")
            st.write("Available inputs:", action.available_inputs or "None")
            st.write("Missing inputs:", action.missing_inputs or "None")
            st.write("Suggested tools:", action.suggested_tools or "None")

    st.subheader("Clarifying questions")
    if response.clarifying_questions:
        for question in response.clarifying_questions:
            st.write(f"- {question}")
    else:
        st.caption("No clarifying questions.")

    st.download_button(
        "Download raw JSON",
        data=response.model_dump_json(indent=2),
        file_name="intent2action_response.json",
        mime="application/json",
    )


st.set_page_config(page_title="intent2action", page_icon="i2a", layout="wide")
st.title("intent2action")
st.caption("Convert text and images into structured action candidates")

left, right = st.columns([1, 2])

with st.sidebar:
    st.subheader("Model provider")
    st.write(f"Base URL: `{settings.model_base_url}`")
    st.write(f"Model: `{settings.model_name}`")
    st.write(f"Vision enabled: `{settings.model_supports_vision}`")

with left:
    input_type = st.radio("Input type", ["Text", "Image"], horizontal=True)
    context = context_fields()
    text_content = ""
    image_file = None
    if input_type == "Text":
        text_content = st.text_area("Text input", height=220)
    else:
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    submitted = st.button("Infer Actions", type="primary")

with right:
    if submitted:
        try:
            pipeline = get_pipeline()
            with st.spinner("Inferring action candidates..."):
                if input_type == "Text":
                    if not text_content.strip():
                        st.error("Text input is required.")
                        st.stop()
                    result = pipeline.infer_from_text(text_content, context)
                else:
                    if image_file is None:
                        st.error("Image upload is required.")
                        st.stop()
                    if not pipeline.settings.model_supports_vision:
                        st.error(
                            "The configured model provider does not have vision support enabled."
                        )
                        st.stop()
                    result = pipeline.infer_from_image(
                        image_bytes=image_file.getvalue(),
                        filename=image_file.name,
                        context=context,
                    )
            render_response(result)
        except OpenAICompatibleClientError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Action inference failed: {exc}")
    else:
        st.info("Enter text or upload an image, add optional context, then infer actions.")

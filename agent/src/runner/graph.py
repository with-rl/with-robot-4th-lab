"""LangGraph-based planner graph for the MLDT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple

# from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import APIStatusError, RateLimitError

from ..common.enums import ModelNames
from ..common.errors import LLMError, RateLimitExceededError
from ..common.logger import get_logger

# from .state import StateSchema

StateCallable = Callable[[Any], Any]
RouterCallable = Callable[[Any], str]

_DEFAULT_TEMPERATURE_ONLY_MODELS: set[ModelNames] = {
    ModelNames.gpt5,
    ModelNames.gpt5mini,
    ModelNames.gpt5nano,
}

logger = get_logger(__name__)


# ! headers
def format_headers(
    model_name: str, header_payload: Dict[str, Any], token_usage: Dict[str, Any]
) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {"model_name": model_name}

    header_parsers = {
        "x-ratelimit-limit-requests": int,
        "x-ratelimit-limit-tokens": int,
        "x-ratelimit-remaining-requests": int,
        "x-ratelimit-remaining-tokens": int,
        # "x-ratelimit-reset-requests": parse_reset,
        # "x-ratelimit-reset-tokens": parse_reset,
    }

    for key, parser in header_parsers.items():
        if key in header_payload:
            formatted[key] = parser(header_payload[key])

    if "total_tokens" in token_usage:
        formatted["total_tokens"] = int(token_usage["total_tokens"])

    return formatted


def extract_headers(
    message: Any | Dict, model_name: str | None = None
) -> Dict[str, Any]:
    if model_name is None:
        model_name = message.get("model_name")
        if not model_name:
            raise ValueError("model_name must be provided or present in the message.")

    metadata = getattr(message, "response_metadata", None) or {}
    headers = metadata.get("headers") if isinstance(metadata, dict) else None
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    header_payload: Dict[str, Any]
    if headers is None:
        header_payload = {}
    elif isinstance(headers, dict):
        header_payload = dict(headers)
    else:
        try:
            header_payload = {str(key): value for key, value in headers.items()}  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            header_payload = {"raw": headers}

    formatted_headers = format_headers(
        model_name=model_name,
        header_payload=header_payload,
        token_usage=token_usage or {},
    )
    return formatted_headers


# ! llm
def _resolve_temperature(
    model_name: ModelNames, temperature: float | None
) -> float | None:
    if temperature is None:
        return None
    if model_name in _DEFAULT_TEMPERATURE_ONLY_MODELS:
        if temperature not in (1, 1.0):
            logger.warning(
                "Model %s only supports its default temperature; ignoring %s",
                model_name.value,
                temperature,
            )
        return None
    return temperature


def _tag_llm_model(llm: Any, model_name: str) -> None:
    setattr(llm, "_registered_model_name", model_name)


def _resolve_llm_model_name(llm: Any) -> str:
    for attr in ("_registered_model_name", "model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    kwargs = getattr(llm, "kwargs", None)
    if isinstance(kwargs, dict):
        value = kwargs.get("model")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _prompt_value_to_input(prompt_value: PromptValue | str) -> Any:
    if hasattr(prompt_value, "to_messages"):
        try:
            return prompt_value.to_messages()
        except NotImplementedError:
            pass
    if hasattr(prompt_value, "to_string"):
        return prompt_value.to_string()
    return prompt_value


@dataclass
class LLMChainResources:
    prompt: PromptTemplate
    llm: Any
    parser: Any | None = None
    format_instructions: str = ""

    @property
    def returns_pydantic(self) -> bool:
        return isinstance(self.parser, PydanticOutputParser)

    def run(self, inputs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        prompt_value = self.prompt.invoke(inputs)
        llm_input = _prompt_value_to_input(prompt_value)
        model_name = _resolve_llm_model_name(self.llm)
        try:
            raw_output = self.llm.invoke(llm_input)
        except RateLimitError as err:
            response = getattr(err, "response", None)
            retry_after = None
            header_payload: Dict[str, Any] = {}
            if response is not None:
                retry_after = getattr(response.headers, "get", lambda *_: None)(
                    "retry-after"
                )
                try:
                    header_payload = dict(response.headers)
                except Exception:
                    header_payload = {"raw": str(response.headers)}
            ratelimit_headers = format_headers(
                model_name=model_name,
                header_payload=header_payload,
                token_usage={},
            )
            logger.error(
                "Rate limit exceeded for model %s: %s", model_name, str(err).strip()
            )
            raise RateLimitExceededError(
                f"Rate limit hit when calling model {model_name}. "
                "Reduce request rate or verify quota/billing.",
                details={
                    "model_name": model_name,
                    "retry_after": retry_after,
                    "ratelimit": ratelimit_headers,
                },
            ) from err
        except APIStatusError as err:
            response = getattr(err, "response", None)
            header_payload: Dict[str, Any] = {}
            if response is not None:
                try:
                    header_payload = dict(response.headers)
                except Exception:
                    header_payload = {"raw": str(response.headers)}
            logger.error(
                "LLM API call failed for model %s with status %s: %s",
                model_name,
                getattr(err, "status_code", "unknown"),
                str(err).strip(),
            )
            raise LLMError(
                f"LLM call failed for model {model_name}",
                details={
                    "model_name": model_name,
                    "status_code": getattr(err, "status_code", None),
                    "headers": header_payload,
                    "error": str(err),
                },
            ) from err
        parsed_output = (
            self.parser.invoke(raw_output) if self.parser is not None else raw_output
        )
        headers = extract_headers(raw_output, model_name=model_name)
        return parsed_output, headers


def _build_llm_chain(
    llm: Any,
    prompt_text: str,
    parser: Any | None = None,
    *,
    skip_parser: bool = False,
) -> LLMChainResources:
    prompt = PromptTemplate.from_template(prompt_text)
    if skip_parser:
        return LLMChainResources(prompt=prompt, llm=llm)

    parser = parser or StrOutputParser()
    format_instructions = (
        parser.get_format_instructions()  # type: ignore[attr-defined]
        if isinstance(parser, PydanticOutputParser)
        else ""
    )
    return LLMChainResources(
        prompt=prompt,
        llm=llm,
        parser=parser,
        format_instructions=format_instructions,
    )


def create_llm(
    model_name: ModelNames,
    temperature: float = 0.0,
    prompt_cache_key: str | None = None,
    bind_tools: bool = False,
):
    model_name_str = model_name.value
    extra_body: Dict[str, Any] | None = None
    if prompt_cache_key:
        extra_body = {"prompt_cache_key": prompt_cache_key}
    resolved_temperature = _resolve_temperature(model_name, temperature)
    llm_kwargs: Dict[str, Any] = {
        "model": model_name_str,
        "include_response_headers": True,
        "max_retries": 5,  # Retry up to 5 times
        "timeout": 60.0,  # 60 second timeout
    }
    if resolved_temperature is not None:
        llm_kwargs["temperature"] = resolved_temperature
    if extra_body:
        llm_kwargs["extra_body"] = extra_body
    llm = ChatOpenAI(**llm_kwargs)
    # if bind_tools:
    # logger.info("Binding tools to LLM model with %s", prompt_cache_key)
    # llm = llm.bind_tools(tools)
    _tag_llm_model(llm, model_name_str)
    return llm


# ! component
def make_normal_node(
    llm,
    *,
    prompt_text: str,
    make_inputs: Callable,
    parser_output=None,
    state_key="history",
    state_append=True,
    node_name="NODE",
    make_outputs: Callable | None = None,
    modify_state: Callable | None = None,
    printout=True,
    skip_parser: bool = False,
) -> StateCallable:

    parser = (
        PydanticOutputParser(pydantic_object=parser_output)
        if parser_output is not None
        else None
    )
    chain_resources = _build_llm_chain(
        llm,
        prompt_text,
        parser=parser,
        skip_parser=skip_parser,
    )

    def node(state):
        logger.info(f"============= {node_name} ==============")
        inputs = make_inputs(state)
        if chain_resources.returns_pydantic:
            inputs["format_instructions"] = chain_resources.format_instructions

        result, _ = chain_resources.run(inputs)
        if make_outputs is not None:
            result = make_outputs(result)

        if chain_resources.returns_pydantic:
            result = result.model_dump()

        if modify_state is not None:
            state = modify_state(state, result)

        if state_append:
            state[state_key].append(result)
        else:
            state[state_key] = result

        if printout:
            logger.info(f"AI Answer:\n{result}\n")

        return state

    return node


def make_decomp_plan_graph(
    state_schema,
    nodes: Dict[str, Any],
    thread_id: str = "decomp_planning",
):
    workflow = StateGraph(state_schema=state_schema)
    # * ============================================================
    workflow.add_node("goal_decomp", nodes["goal_decomp"])
    workflow.add_node("task_decomp", nodes["task_decomp"])

    # * ============================================================
    workflow.add_edge(START, "goal_decomp")
    workflow.add_edge("goal_decomp", "task_decomp")
    workflow.add_edge("task_decomp", END)

    # memory = MemorySaver()
    # graph = workflow.compile(checkpointer=memory)
    graph = workflow.compile(checkpointer=None)
    config = {"configurable": {"thread_id": thread_id}}
    return graph, config

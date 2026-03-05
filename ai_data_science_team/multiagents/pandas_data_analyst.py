from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated, Sequence, Union

import pandas as pd
import json
from IPython.display import Markdown

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.agents import DataWranglingAgent, DataVisualizationAgent
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team.utils.regex import (
    remove_consecutive_duplicates,
    get_generic_summary,
)

AGENT_NAME = "pandas_data_analyst"


class PandasDataAnalyst(BaseAgent):
    """
    PandasDataAnalyst is a multi-agent class that combines data wrangling and visualization capabilities.
    The orchestrator LLM uses tool-calling to invoke sub-agents rather than hard-coded routing.

    Parameters:
    -----------
    model:
        The language model to be used for the agents.
    data_wrangling_agent: DataWranglingAgent
        The Data Wrangling Agent for transforming raw data.
    data_visualization_agent: DataVisualizationAgent
        The Data Visualization Agent for generating plots.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.

    Methods:
    --------
    ainvoke_agent(user_instructions, data_raw, **kwargs)
        Asynchronously invokes the multi-agent with user instructions and raw data.
    invoke_agent(user_instructions, data_raw, **kwargs)
        Synchronously invokes the multi-agent with user instructions and raw data.
    get_data_wrangled()
        Returns the wrangled data as a Pandas DataFrame.
    get_plotly_graph()
        Returns the Plotly graph as a Plotly object.
    get_data_wrangler_function(markdown=False)
        Returns the data wrangling function as a string, optionally in Markdown.
    get_data_visualization_function(markdown=False)
        Returns the data visualization function as a string, optionally in Markdown.
    """

    def __init__(
        self,
        model,
        data_wrangling_agent: DataWranglingAgent,
        data_visualization_agent: DataVisualizationAgent,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_visualization_agent": data_visualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_pandas_data_analyst(
            model=self._params["model"],
            data_wrangling_agent=self._params["data_wrangling_agent"]._compiled_graph,
            data_visualization_agent=self._params[
                "data_visualization_agent"
            ]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        user_instructions,
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Asynchronously invokes the multi-agent."""
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response

    def invoke_agent(
        self,
        user_instructions,
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Synchronously invokes the multi-agent."""
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Invoke the multi-agent with an explicit message list (preferred for supervisors/teams).
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    user_instructions = msg.content
                    break
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return None

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Async version of invoke_messages.
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    user_instructions = msg.content
                    break
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return None

    def get_data_wrangled(self):
        """Returns the wrangled data as a Pandas DataFrame."""
        if self.response and self.response.get("data_wrangled"):
            return pd.DataFrame(self.response.get("data_wrangled"))

    def get_plotly_graph(self):
        """Returns the Plotly graph as a Plotly object."""
        if self.response and self.response.get("plotly_graph"):
            return plotly_from_dict(self.response.get("plotly_graph"))

    def get_data_wrangler_function(self, markdown=False):
        """Returns the data wrangling function as a string."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response.get("data_wrangler_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_data_visualization_function(self, markdown=False):
        """Returns the data visualization function as a string."""
        if self.response and self.response.get("data_visualization_function"):
            code = self.response.get("data_visualization_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_workflow_summary(self, markdown=False):
        """Returns a summary of the workflow."""
        if self.response and self.response.get("messages"):
            agents = []
            seen = set()
            # Only list sub-agents (exclude assistant/system/user)
            allowed = {"data_wrangling_agent", "data_visualization_agent"}
            role_to_content = {}
            for msg in self.response["messages"]:
                role = getattr(msg, "role", None) or getattr(msg, "type", None)
                if role in allowed and role not in seen:
                    agents.append(role)
                    seen.add(role)
                if role in allowed and role not in role_to_content:
                    role_to_content[role] = getattr(msg, "content", "")
            agent_labels = [
                f"- **Agent {i + 1}:** {role}\n" for i, role in enumerate(agents)
            ]
            header = (
                f"# Pandas Data Analyst Workflow Summary\n\nThis workflow contains {len(agents)} agents:\n\n"
                + "\n".join(agent_labels)
            )
            reports = []
            for role in agents:
                content = role_to_content.get(role, "")
                try:
                    reports.append(get_generic_summary(json.loads(content)))
                except Exception:
                    reports.append(content)
            summary = "\n\n" + header + "\n\n".join(reports)
            return Markdown(summary) if markdown else summary

    @staticmethod
    def _convert_data_input(
        data_raw: Union[pd.DataFrame, dict, list],
    ) -> Union[dict, list]:
        """Converts input data to the expected format (dict or list of dicts)."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        if isinstance(data_raw, dict):
            return data_raw
        if isinstance(data_raw, list):
            return [
                item.to_dict() if isinstance(item, pd.DataFrame) else item
                for item in data_raw
            ]
        raise ValueError(
            "data_raw must be a DataFrame, dict, or list of DataFrames/dicts"
        )


def make_pandas_data_analyst(
    model,
    data_wrangling_agent: CompiledStateGraph,
    data_visualization_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None,
):
    """
    Creates a multi-agent system that wrangles data and optionally visualizes it.
    The orchestrator LLM uses tool-calling to decide which sub-agents to invoke,
    replacing hard-coded routing logic.

    Parameters:
    -----------
    model: The language model to be used.
    data_wrangling_agent: CompiledStateGraph
        The Data Wrangling Agent.
    data_visualization_agent: CompiledStateGraph
        The Data Visualization Agent.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state.

    Returns:
    --------
    CompiledStateGraph: The compiled multi-agent system.
    """

    llm = model

    # --- Tool definitions (schema only; actual invocation happens in execute_tools) ---

    @tool
    def data_wrangling_tool(user_instructions: str) -> str:
        """
        Wrangle, transform, filter, aggregate, or reshape pandas data based on user instructions.
        Always call this tool first before any visualization.
        """
        return user_instructions

    @tool
    def data_visualization_tool(user_instructions: str) -> str:
        """
        Create a Plotly chart or visualization based on user instructions.
        Only call this tool when the user explicitly requests a chart or plot.
        """
        return user_instructions

    tools = [data_wrangling_tool, data_visualization_tool]
    llm_with_tools = llm.bind_tools(tools)

    ORCHESTRATOR_SYSTEM_PROMPT = (
        "You are a pandas data analyst orchestrator. You have two tools:\n"
        "1. `data_wrangling_tool`: Transforms, filters, aggregates, or reshapes data. "
        "Always call this first with the data manipulation part of the user's request.\n"
        "2. `data_visualization_tool`: Creates a Plotly chart. "
        "Call this only if the user explicitly asks for a chart or visualization.\n\n"
        "Rules:\n"
        "- Always call `data_wrangling_tool` exactly once.\n"
        "- Call `data_visualization_tool` at most once, and only if a chart is requested.\n"
        "- After all tool calls are complete, provide a brief summary of what was done."
    )

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_instructions: str
        data_raw: Union[dict, list]
        data_wrangled: dict
        data_wrangler_function: str
        data_visualization_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int

    def prepare_messages(state: PrimaryState):
        print("---PANDAS DATA ANALYST---")
        print("*************************")
        print("---PREPARE MESSAGES---")
        msgs = list(state.get("messages", []))
        ui = state.get("user_instructions")

        # Inject system prompt if not already present
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)] + msgs

        if not msgs or (len(msgs) == 1 and isinstance(msgs[0], SystemMessage)):
            msgs.append(HumanMessage(content=ui))

        if not ui:
            for msg in reversed(msgs):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    ui = msg.content
                    break

        normalized = []
        for msg in msgs:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in ("user", "human"):
                    normalized.append(HumanMessage(content=content))
                elif role == "system":
                    normalized.append(SystemMessage(content=content))
                elif role in ("assistant", "ai"):
                    normalized.append(AIMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=str(content)))
            else:
                normalized.append(HumanMessage(content=str(msg)))

        return {"messages": normalized, "user_instructions": ui}

    def orchestrator_node(state: PrimaryState):
        print("---ORCHESTRATOR (LLM with Tools)---")
        messages = state.get("messages", [])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def execute_tools(state: PrimaryState):
        print("---EXECUTE TOOLS---")
        messages = state.get("messages", [])
        last_message = messages[-1]

        updates = {}
        new_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "data_wrangling_tool":
                print("---INVOKING DATA WRANGLING AGENT---")
                response = data_wrangling_agent.invoke(
                    {
                        "user_instructions": tool_args.get("user_instructions"),
                        "data_raw": state.get("data_raw"),
                        "max_retries": state.get("max_retries"),
                        "retry_count": state.get("retry_count"),
                    }
                )
                data_wrangled = response.get("data_wrangled")
                updates["data_wrangled"] = data_wrangled
                updates["data_wrangler_function"] = response.get("data_wrangler_function")
                new_messages.extend(response.get("messages", []))

                try:
                    shape = pd.DataFrame(data_wrangled).shape if data_wrangled else None
                    result_summary = (
                        f"Data wrangling complete. Result shape: {shape[0]} rows x {shape[1]} cols."
                        if shape
                        else "Data wrangling complete."
                    )
                except Exception:
                    result_summary = "Data wrangling complete."

                new_messages.append(
                    ToolMessage(content=result_summary, tool_call_id=tool_call_id)
                )

            elif tool_name == "data_visualization_tool":
                print("---INVOKING DATA VISUALIZATION AGENT---")
                data_for_viz = (
                    updates.get("data_wrangled")
                    or state.get("data_wrangled")
                    or state.get("data_raw")
                )
                if data_for_viz is None:
                    error_msg = "No data available for visualization; skipped."
                    updates["plotly_error"] = error_msg
                    new_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    )
                    continue

                response = data_visualization_agent.invoke(
                    {
                        "user_instructions": tool_args.get("user_instructions"),
                        "data_raw": data_for_viz,
                        "max_retries": state.get("max_retries"),
                        "retry_count": state.get("retry_count"),
                    }
                )
                updates["plotly_graph"] = response.get("plotly_graph")
                updates["data_visualization_function"] = response.get("data_visualization_function")
                updates["plotly_error"] = response.get("data_visualization_error")
                new_messages.extend(response.get("messages", []))

                result_summary = (
                    "Chart created successfully."
                    if response.get("plotly_graph")
                    else f"Chart creation failed: {response.get('data_visualization_error', 'Unknown error')}"
                )
                new_messages.append(
                    ToolMessage(content=result_summary, tool_call_id=tool_call_id)
                )

        return {"messages": new_messages, **updates}

    def should_continue(state: PrimaryState):
        last_message = state.get("messages", [])[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "finalize"

    def finalize_output(state: PrimaryState):
        print("---FINALIZE OUTPUT---")
        data_wrangled = state.get("data_wrangled")
        plot = state.get("plotly_graph")
        plot_err = state.get("plotly_error")
        parts = []
        if data_wrangled:
            try:
                df = pd.DataFrame(data_wrangled)
                parts.append(f"Wrangled table shape: {df.shape[0]} rows x {df.shape[1]} cols.")
            except Exception:
                parts.append("Wrangled data available.")
        if plot:
            parts.append("Chart created from wrangled data.")
        elif plot_err:
            parts.append(f"Chart not created: {plot_err}")
        summary = " ".join(parts) or "Workflow completed."
        ai_msg = AIMessage(content=summary, role="assistant")
        return {"messages": [ai_msg]}

    workflow = StateGraph(PrimaryState)

    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("finalize_output", finalize_output)

    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {"tools": "tools", "finalize": "finalize_output"},
    )
    workflow.add_edge("tools", "orchestrator")
    workflow.add_edge("finalize_output", END)

    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)

    return app

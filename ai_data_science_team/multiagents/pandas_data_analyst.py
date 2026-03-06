from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool as langchain_tool
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated, Sequence, Union
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ---------------------------------------------------------------------------
# PandasDataAnalyst class
# ---------------------------------------------------------------------------

class PandasDataAnalyst(BaseAgent):
    """
    PandasDataAnalyst is a multi-agent class that combines data wrangling and
    visualization capabilities using a flexible LLM-based task planner.

    The planner dynamically generates a task DAG (directed acyclic graph) from
    the user's instructions.  Independent tasks in the DAG are executed in
    parallel, reducing latency for multi-chart or multi-dataset requests.

    Parameters
    ----------
    model :
        The language model used for planning and the sub-agents.
    data_wrangling_agent : DataWranglingAgent
        The Data Wrangling Agent for transforming raw data.
    data_visualization_agent : DataVisualizationAgent
        The Data Visualization Agent for generating Plotly charts.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Methods
    -------
    ainvoke_agent(user_instructions, data_raw, **kwargs)
        Asynchronously invokes the multi-agent.
    invoke_agent(user_instructions, data_raw, **kwargs)
        Synchronously invokes the multi-agent.
    get_data_wrangled()
        Returns the primary wrangled DataFrame.
    get_plotly_graph()
        Returns the first/primary Plotly figure.
    get_plotly_graphs()
        Returns all Plotly figures produced (for parallel chart tasks).
    get_data_wrangler_function(markdown=False)
        Returns the generated wrangling function code.
    get_data_visualization_function(markdown=False)
        Returns the generated visualization function code.
    get_task_plan()
        Returns the LLM-generated task plan as a list of dicts.
    get_workflow_summary(markdown=False)
        Returns a Markdown summary of the completed workflow.
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
        Invoke the multi-agent with an explicit message list
        (preferred for supervisors / team workflows).
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if (
                    getattr(msg, "type", None) == "human"
                    or getattr(msg, "role", None) == "user"
                ):
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
        """Async version of invoke_messages."""
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if (
                    getattr(msg, "type", None) == "human"
                    or getattr(msg, "role", None) == "user"
                ):
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

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_data_wrangled(self) -> Optional[pd.DataFrame]:
        """Returns the primary wrangled DataFrame (last wrangling task)."""
        if self.response and self.response.get("data_wrangled"):
            return pd.DataFrame(self.response["data_wrangled"])
        return None

    def get_plotly_graph(self):
        """Returns the first/primary Plotly figure."""
        if self.response and self.response.get("plotly_graph"):
            return plotly_from_dict(self.response["plotly_graph"])
        return None

    def get_plotly_graphs(self) -> list:
        """
        Returns all Plotly figures produced by the workflow.

        Useful when the planner creates multiple parallel visualization tasks
        (e.g., one chart per model or multiple chart types).
        """
        if self.response and self.response.get("plotly_graphs"):
            return [plotly_from_dict(g) for g in self.response["plotly_graphs"] if g]
        return []

    def get_data_wrangler_function(self, markdown: bool = False):
        """Returns the primary data-wrangling function code."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response["data_wrangler_function"]
            return Markdown(f"```python\n{code}\n```") if markdown else code
        return None

    def get_data_visualization_function(self, markdown: bool = False):
        """Returns the primary data-visualization function code."""
        if self.response and self.response.get("data_visualization_function"):
            code = self.response["data_visualization_function"]
            return Markdown(f"```python\n{code}\n```") if markdown else code
        return None

    def get_task_plan(self) -> list:
        """Returns the LLM-generated task plan (list of task dicts)."""
        if self.response:
            return self.response.get("task_plan", [])
        return []

    def get_workflow_summary(self, markdown: bool = False):
        """Returns a summary of the completed workflow."""
        if not (self.response and self.response.get("messages")):
            return None

        agents = []
        seen: set = set()
        allowed = {"data_wrangling_agent", "data_visualization_agent"}
        role_to_content: dict = {}

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

        # Task plan section
        task_plan = self.get_task_plan()
        plan_lines = []
        if task_plan:
            plan_lines.append(f"\n**Task plan ({len(task_plan)} tasks):**\n")
            for t in task_plan:
                deps = ", ".join(t.get("depends_on", [])) or "none"
                plan_lines.append(
                    f"  - `{t['task_id']}` [{t['tool']}] (depends: {deps})\n"
                )

        header = (
            f"# Pandas Data Analyst Workflow Summary\n\n"
            f"This workflow contains {len(agents)} agent(s):\n\n"
            + "".join(agent_labels)
            + "".join(plan_lines)
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
        """Converts input data to dict or list-of-dicts."""
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


# ---------------------------------------------------------------------------
# Module-level helpers (data summary, task execution, parallel runner)
# ---------------------------------------------------------------------------

def _make_data_summary(
    data_raw: Union[dict, list],
    max_cols: int = 25,
    max_unique: int = 8,
) -> str:
    """
    Build a lightweight string summary of the raw data for the planner LLM.

    Includes shape, column names/types, and sample categorical values so the
    planner can make informed decisions about task splitting.
    """
    try:
        if isinstance(data_raw, dict):
            df = pd.DataFrame(data_raw)
            lines = [f"Single dataset: {df.shape[0]} rows x {df.shape[1]} columns"]
            for col in df.columns[:max_cols]:
                dtype = str(df[col].dtype)
                if df[col].dtype == object or str(df[col].dtype) == "category":
                    vals = df[col].dropna().unique()[:max_unique].tolist()
                    more = "…" if df[col].nunique() > max_unique else ""
                    lines.append(f"  - {col} (string): sample = {vals}{more}")
                else:
                    lines.append(
                        f"  - {col} ({dtype}): range [{df[col].min()} .. {df[col].max()}]"
                    )
            return "\n".join(lines)

        if isinstance(data_raw, list):
            parts = []
            for i, d in enumerate(data_raw):
                df = pd.DataFrame(d)
                cols = list(df.columns[:10])
                parts.append(
                    f"Dataset {i + 1}: {df.shape[0]} rows x {df.shape[1]} cols, "
                    f"columns: {cols}"
                )
            return "\n".join(parts)

    except Exception as exc:
        return f"(Data summary unavailable: {exc})"

    return "Data format unknown"


def _execute_single_task(
    task: dict,
    upstream_results: dict,
    tool_map: dict,
    data_raw: Union[dict, list],
) -> tuple:
    """
    Execute one task from the plan and return ``(task_id, result_dict)``.

    Parameters
    ----------
    task : dict
        TaskSpec with keys: task_id, tool, instructions, depends_on, input_from.
    upstream_results : dict
        Already-completed results keyed by task_id.
    tool_map : dict
        Mapping of tool name → LangChain tool object.
    data_raw : dict or list
        Original input data (fallback when ``input_from`` is absent).
    """
    task_id = task["task_id"]
    tool_name = task["tool"]
    instructions = task["instructions"]
    input_from = task.get("input_from")

    # Determine which data to feed into this task
    if input_from and input_from in upstream_results:
        data_in = upstream_results[input_from].get("data_wrangled") or data_raw
    else:
        data_in = data_raw

    try:
        if tool_name == "data_wrangling":
            result = tool_map["data_wrangling"].invoke(
                {"instructions": instructions, "data_raw": data_in}
            )
        elif tool_name == "data_visualization":
            result = tool_map["data_visualization"].invoke(
                {"instructions": instructions, "data_wrangled": data_in}
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}", "messages": []}
    except Exception as exc:
        result = {"error": str(exc), "messages": []}

    return task_id, result


def _run_plan_parallel(
    task_plan: list,
    tool_map: dict,
    data_raw: Union[dict, list],
) -> dict:
    """
    Execute a task plan respecting dependency order while running
    independent tasks in parallel using threads.

    Each iteration finds all tasks whose dependencies are satisfied
    and dispatches them concurrently via ``ThreadPoolExecutor``.

    Parameters
    ----------
    task_plan : list of TaskSpec dicts
    tool_map : dict  (tool_name -> LangChain tool)
    data_raw : original input data

    Returns
    -------
    dict mapping task_id -> result dict
    """
    completed: dict = {}
    remaining = list(task_plan)

    while remaining:
        ready = [
            t
            for t in remaining
            if all(dep in completed for dep in t.get("depends_on", []))
        ]

        if not ready:
            # Guard against circular/unresolvable dependencies
            unresolved = [t["task_id"] for t in remaining]
            print(f"    WARNING: could not resolve tasks {unresolved}; skipping.")
            break

        if len(ready) == 1:
            task_id, result = _execute_single_task(
                ready[0], completed, tool_map, data_raw
            )
            completed[task_id] = result
        else:
            # Parallel batch — pass a read-only snapshot of completed so each
            # thread sees a consistent view of upstream results.
            snapshot = dict(completed)
            with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                futures = {
                    pool.submit(
                        _execute_single_task, task, snapshot, tool_map, data_raw
                    ): task["task_id"]
                    for task in ready
                }
                for future in as_completed(futures):
                    task_id, result = future.result()
                    completed[task_id] = result

        remaining = [t for t in remaining if t["task_id"] not in completed]

    return completed


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def make_pandas_data_analyst(
    model,
    data_wrangling_agent: CompiledStateGraph,
    data_visualization_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None,
):
    """
    Creates a flexible multi-agent system for pandas-based data analysis.

    Architecture
    ------------
    START → prepare_messages → plan_tasks → execute_tasks → finalize_output → END

    * **plan_tasks** – An LLM planner reads the user instructions and data
      summary and outputs a task DAG (list of TaskSpec dicts with dependency
      information).  This replaces the old hard-coded router.

    * **execute_tasks** – Walks the DAG, running independent tasks in parallel
      threads.  Subagents are invoked through LangChain tools
      (``data_wrangling_tool`` / ``data_visualization_tool``), which wrap the
      compiled LangGraph sub-agents.

    * **finalize_output** – Aggregates results, sets backward-compatible state
      fields, and appends a summary message.

    Parallelism examples
    --------------------
    * "Show bar charts for model A and model B" →
      [wrangle_A, wrangle_B] run in parallel, then
      [viz_A, viz_B] run in parallel.

    * "Show scatter plot and histogram for model A" →
      wrangle_A runs first, then
      [viz_scatter, viz_hist] run in parallel.

    Parameters
    ----------
    model : LLM
    data_wrangling_agent : CompiledStateGraph
    data_visualization_agent : CompiledStateGraph
    checkpointer : Checkpointer, optional

    Returns
    -------
    CompiledStateGraph
    """
    llm = model

    # ------------------------------------------------------------------
    # LangChain tools wrapping each subagent
    # ------------------------------------------------------------------

    @langchain_tool
    def data_wrangling_tool(instructions: str, data_raw: dict) -> dict:
        """
        Wrangle and transform data using the Data Wrangling Agent.

        Apply pandas-based operations such as filtering, aggregating,
        reshaping, merging, pivoting, encoding, or computing new columns.
        Returns the wrangled data dict and the generated Python function.

        Args:
            instructions: Specific wrangling instructions for this task.
            data_raw: Source data as a dict mapping column names to value lists.

        Returns:
            dict with keys: data_wrangled, data_wrangler_function, messages.
        """
        response = data_wrangling_agent.invoke(
            {
                "user_instructions": instructions,
                "data_raw": data_raw,
                "max_retries": 3,
                "retry_count": 0,
            }
        )
        return {
            "data_wrangled": response.get("data_wrangled"),
            "data_wrangler_function": response.get("data_wrangler_function"),
            "messages": response.get("messages", []),
        }

    @langchain_tool
    def data_visualization_tool(instructions: str, data_wrangled: dict) -> dict:
        """
        Create an interactive Plotly visualization using the Data Visualization Agent.

        Generate charts such as bar, line, scatter, histogram, box, heatmap, etc.
        Returns the Plotly figure dict and the generated Python function.

        Args:
            instructions: Visualization instructions (chart type, axes, title, style).
            data_wrangled: Prepared data to visualize as a dict.

        Returns:
            dict with keys: plotly_graph, data_visualization_function, messages.
        """
        response = data_visualization_agent.invoke(
            {
                "user_instructions": instructions,
                "data_raw": data_wrangled,
                "max_retries": 3,
                "retry_count": 0,
            }
        )
        return {
            "plotly_graph": response.get("plotly_graph"),
            "data_visualization_function": response.get("data_visualization_function"),
            "messages": response.get("messages", []),
        }

    tool_map = {
        "data_wrangling": data_wrangling_tool,
        "data_visualization": data_visualization_tool,
    }

    # ------------------------------------------------------------------
    # Planner prompt & chain
    # ------------------------------------------------------------------

    _PLANNER_PROMPT = PromptTemplate(
        template="""You are a task planner for a Pandas Data Analysis system.

Given the user's request and the data summary below, produce a task plan that
specifies which agents to call and in what dependency order.

=== AVAILABLE TOOLS ===
- "data_wrangling"    : filters, aggregates, reshapes, or transforms data with pandas.
                        Takes `data_raw` as input.  Returns a table (data_wrangled).
- "data_visualization": creates interactive Plotly charts from data.
                        Must always consume output from a preceding wrangling task.

=== PLANNING RULES ===
1. Always include at least one "data_wrangling" task.
2. Every "data_visualization" task MUST list a "data_wrangling" task_id in both
   "depends_on" AND "input_from".
3. MULTIPLE SEPARATE ANALYSES (different models, categories, or datasets):
   - Create INDEPENDENT parallel pipelines — each with its own wrangle + viz tasks.
   - Independent tasks have no overlapping "depends_on" entries.
   - Example: wrangle_A (deps:[]) → viz_A (deps:[wrangle_A])
              wrangle_B (deps:[]) → viz_B (deps:[wrangle_B])
     wrangle_A and wrangle_B execute in parallel; viz_A and viz_B execute in parallel.
4. ONE DATASET with MULTIPLE CHARTS (same data, different chart types):
   - Create ONE "data_wrangling" task, then MULTIPLE "data_visualization" tasks
     that all depend on it.
   - Example: wrangle_1 (deps:[]) → viz_scatter (deps:[wrangle_1])
                                   → viz_hist    (deps:[wrangle_1])
     viz_scatter and viz_hist execute in parallel after wrangle_1 finishes.
5. If NO visualization is requested, use only "data_wrangling" tasks.
6. task_id must be short, snake_case, and descriptive
   (e.g., "wrangle_model_a", "viz_bar_revenue").
7. "instructions" must be self-contained and specific to that task only.
8. "input_from" is null for wrangling tasks (they always receive data_raw).
   For visualization tasks, "input_from" is the task_id of their data source.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object — no markdown fences, no explanation text.

{{
  "tasks": [
    {{
      "task_id": "<string>",
      "tool": "data_wrangling" | "data_visualization",
      "instructions": "<specific instructions for this task>",
      "depends_on": [],
      "input_from": null | "<wrangling_task_id>"
    }}
  ]
}}

=== EXAMPLES ===

User: "Show bar charts for model A and line charts for model B"
{{
  "tasks": [
    {{"task_id": "wrangle_model_a", "tool": "data_wrangling",
      "instructions": "Filter data for model A only.", "depends_on": [], "input_from": null}},
    {{"task_id": "wrangle_model_b", "tool": "data_wrangling",
      "instructions": "Filter data for model B only.", "depends_on": [], "input_from": null}},
    {{"task_id": "viz_bar_model_a", "tool": "data_visualization",
      "instructions": "Create a bar chart for model A.", "depends_on": ["wrangle_model_a"], "input_from": "wrangle_model_a"}},
    {{"task_id": "viz_line_model_b", "tool": "data_visualization",
      "instructions": "Create a line chart for model B.", "depends_on": ["wrangle_model_b"], "input_from": "wrangle_model_b"}}
  ]
}}

User: "For model A, show a scatter plot of price vs mileage and a histogram of price"
{{
  "tasks": [
    {{"task_id": "wrangle_model_a", "tool": "data_wrangling",
      "instructions": "Filter data for model A only.", "depends_on": [], "input_from": null}},
    {{"task_id": "viz_scatter_price_mileage", "tool": "data_visualization",
      "instructions": "Create a scatter plot of price (y) vs mileage (x) for model A.", "depends_on": ["wrangle_model_a"], "input_from": "wrangle_model_a"}},
    {{"task_id": "viz_histogram_price", "tool": "data_visualization",
      "instructions": "Create a histogram of the price column for model A.", "depends_on": ["wrangle_model_a"], "input_from": "wrangle_model_a"}}
  ]
}}

User: "Show the top 10 rows sorted by sales descending"
{{
  "tasks": [
    {{"task_id": "wrangle_top10", "tool": "data_wrangling",
      "instructions": "Sort by sales descending and return the top 10 rows.", "depends_on": [], "input_from": null}}
  ]
}}

=== DATA SUMMARY ===
{data_summary}

=== USER INSTRUCTIONS ===
{user_instructions}
""",
        input_variables=["user_instructions", "data_summary"],
    )

    _planner_chain = _PLANNER_PROMPT | llm | JsonOutputParser()

    # ------------------------------------------------------------------
    # LangGraph state
    # ------------------------------------------------------------------

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_instructions: str
        data_raw: Union[dict, list]
        # Planner output
        task_plan: list          # List of TaskSpec dicts
        task_results: dict       # task_id → result dict
        # Backward-compatible result fields
        data_wrangled: dict
        data_wrangler_function: str
        data_visualization_function: str
        plotly_graph: dict       # First/primary chart
        plotly_graphs: list      # All charts (for parallel viz tasks)
        plotly_error: str
        max_retries: int
        retry_count: int

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def prepare_messages(state: PrimaryState):
        print("---PANDAS DATA ANALYST---")
        print("*************************")
        print("---PREPARE MESSAGES---")
        msgs = state.get("messages", [])
        ui = state.get("user_instructions")
        if not msgs:
            system_hint = (
                "You are a pandas data analyst orchestrator. Route the user's "
                "question to data wrangling and optional visualization. Prefer "
                "tables unless the user clearly requests a chart."
            )
            msgs = [("system", system_hint), ("user", ui)]
        if not ui:
            for msg in reversed(msgs):
                if (
                    getattr(msg, "type", None) == "human"
                    or getattr(msg, "role", None) == "user"
                ):
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

    def plan_tasks(state: PrimaryState):
        """
        LLM planner: converts user instructions into a task DAG.

        The planner is given a lightweight data summary so it can make
        intelligent decisions about parallel pipelines (e.g., one branch per
        model) vs. fan-out after a single wrangling step.
        """
        print("---PLAN TASKS (LLM PLANNER)---")
        user_instructions = state.get("user_instructions", "")
        data_raw = state.get("data_raw")
        data_summary = _make_data_summary(data_raw)

        try:
            plan = _planner_chain.invoke(
                {
                    "user_instructions": user_instructions,
                    "data_summary": data_summary,
                }
            )
            task_plan = plan.get("tasks", [])
            if not isinstance(task_plan, list) or not task_plan:
                raise ValueError("Planner returned an empty task list.")
        except Exception as exc:
            print(f"    Planner failed ({exc}). Using fallback single-task plan.")
            task_plan = [
                {
                    "task_id": "wrangle_main",
                    "tool": "data_wrangling",
                    "instructions": user_instructions,
                    "depends_on": [],
                    "input_from": None,
                }
            ]

        ids = [t["task_id"] for t in task_plan]
        print(f"    Task plan: {ids}")
        return {"task_plan": task_plan, "task_results": {}}

    def execute_tasks(state: PrimaryState):
        """
        Execute the task plan with parallel execution of independent tasks.

        Tasks with no unmet dependencies run concurrently in threads.
        Each task invokes its corresponding LangChain tool, which delegates
        to the appropriate compiled subagent graph.
        """
        print("---EXECUTE TASKS (PARALLEL WHERE POSSIBLE)---")
        task_plan = state.get("task_plan", [])
        data_raw = state.get("data_raw")

        results = _run_plan_parallel(task_plan, tool_map, data_raw)
        print(f"    Completed {len(results)}/{len(task_plan)} tasks.")
        return {"task_results": results}

    def finalize_output(state: PrimaryState):
        """
        Aggregate task results into backward-compatible state fields and
        append a workflow-summary AI message.
        """
        print("---FINALIZE OUTPUT---")
        task_plan = state.get("task_plan", [])
        task_results = state.get("task_results", {})

        wrangle_results: list = []
        viz_results: list = []
        all_agent_messages: list = []

        for task in task_plan:
            task_id = task["task_id"]
            result = task_results.get(task_id, {})
            all_agent_messages.extend(result.get("messages", []))
            if task["tool"] == "data_wrangling" and result.get("data_wrangled"):
                wrangle_results.append(result)
            elif task["tool"] == "data_visualization" and result.get("plotly_graph"):
                viz_results.append(result)

        # Primary (backward-compat) fields
        data_wrangled = wrangle_results[-1]["data_wrangled"] if wrangle_results else None
        data_wrangler_function = (
            wrangle_results[-1].get("data_wrangler_function") if wrangle_results else None
        )
        plotly_graph = viz_results[0]["plotly_graph"] if viz_results else None
        plotly_graphs = [r["plotly_graph"] for r in viz_results]
        data_visualization_function = (
            viz_results[0].get("data_visualization_function") if viz_results else None
        )

        # Summary message
        parts = []
        if wrangle_results:
            try:
                df = pd.DataFrame(data_wrangled)
                parts.append(
                    f"Wrangled table shape: {df.shape[0]} rows x {df.shape[1]} cols."
                )
            except Exception:
                parts.append("Data wrangling completed.")
        if viz_results:
            parts.append(f"{len(viz_results)} chart(s) created.")
        if not wrangle_results and not viz_results:
            parts.append("Workflow completed with no output.")

        ai_msg = AIMessage(content=" ".join(parts), role="assistant")

        return {
            "messages": all_agent_messages + [ai_msg],
            "data_wrangled": data_wrangled,
            "data_wrangler_function": data_wrangler_function,
            "data_visualization_function": data_visualization_function,
            "plotly_graph": plotly_graph,
            "plotly_graphs": plotly_graphs,
        }

    # ------------------------------------------------------------------
    # Assemble the graph
    # ------------------------------------------------------------------

    workflow = StateGraph(PrimaryState)

    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("plan_tasks", plan_tasks)
    workflow.add_node("execute_tasks", execute_tasks)
    workflow.add_node("finalize_output", finalize_output)

    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "plan_tasks")
    workflow.add_edge("plan_tasks", "execute_tasks")
    workflow.add_edge("execute_tasks", "finalize_output")
    workflow.add_edge("finalize_output", END)

    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    return app

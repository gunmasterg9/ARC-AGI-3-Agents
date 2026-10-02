from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .arcagi_solver import ARCAGI3Solver
from .arcagi_v3 import ARCAGIV3Solver
from .recorder import Recorder
from .swarm import Swarm
try:
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
    from .templates.langgraph_random_agent import LangGraphRandom
    from .templates.langgraph_thinking import LangGraphThinking
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
    from .templates.multimodal import MultiModalLLM
    from .templates.random_agent import Random
    from .templates.reasoning_agent import ReasoningAgent
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
except ImportError:
    pass

load_dotenv()


AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# Add the custom ARC-AGI-3 solver (V2).
AVAILABLE_AGENTS["arcagi3solver"] = ARCAGI3Solver
AVAILABLE_AGENTS["arcagi_solver"] = ARCAGI3Solver
AVAILABLE_AGENTS["arcagisolver"] = ARCAGI3Solver

# Add the ARC-AGI-3 V3 general solver.
AVAILABLE_AGENTS["arcagiv3"] = ARCAGIV3Solver
AVAILABLE_AGENTS["arcagi_v3"] = ARCAGIV3Solver
AVAILABLE_AGENTS["arcagi3_v3"] = ARCAGIV3Solver
AVAILABLE_AGENTS["arcagiv3solver"] = ARCAGIV3Solver

# Add all recording files as valid agent names.
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# Add subclasses that aren't discovered directly through Agent.__subclasses__().
if "ReasoningAgent" in locals():
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent


__all__ = [
    "ARCAGI3Solver",
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
]
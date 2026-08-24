import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.agent import SupportAgent
from app.retrieval.index import KnowledgeIndex

def test():
    storage_dir = _PROJECT_ROOT / "storage"
    index = KnowledgeIndex.load(storage_dir)
    agent = SupportAgent(index)

    queries = [
        "What is the standard return window?",
        "Do you ship internationally?",
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        "Are all your fabrics vegan?",
        "Where is ORD-1007?",
        "When will it arrive?",
        "Where is my order?",
        "When will ORD-1004 arrive?"
    ]

    for q in queries:
        print(f"\n======================================")
        print(f"Q: {q}")
        res = agent.chat(q)
        print(f"\nA: {res.answer}")
        print(f"Sources: {res.sources}")
        print(f"Handoff: {res.handoff}")

if __name__ == "__main__":
    test()

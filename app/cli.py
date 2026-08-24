import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.agent import SupportAgent
from app.retrieval.index import KnowledgeIndex

def main():
    storage_dir = _PROJECT_ROOT / "storage"
    if not storage_dir.exists() or not (storage_dir / "chunks.json").exists():
        print("[!] Storage index not found. Building index first...")
        from scripts.build_index import build_index
        build_index(_PROJECT_ROOT / "knowledge-base", storage_dir, verbose=False)

    try:
        index = KnowledgeIndex.load(storage_dir)
    except Exception as e:
        print(f"Failed to load index: {e}")
        return

    agent = SupportAgent(index)
    print("Aster & Row Support Agent")
    print("Type 'exit' or 'quit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue

            response = agent.chat(user_input)
            
            print(f"\nAgent:\n{response.answer}\n")
            if response.sources:
                print("Sources:")
                for src in response.sources:
                    print(f"- {src}")
                print()
            
            print(f"Human handoff: {'Yes' if response.handoff else 'No'}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Error: {e}\n")

if __name__ == "__main__":
    main()

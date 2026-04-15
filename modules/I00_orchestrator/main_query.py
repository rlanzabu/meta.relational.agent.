import sys
from pathlib import Path

# Add modules directory to path so we can import sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from G00_retriever import Retriever, NoResultsFound

def main_query():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        context_text, results = Retriever.retriever(query_text)
        Retriever.print_results(results, context_text, query_text)
    except NoResultsFound as e:
        print(f"❌ {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main_query())
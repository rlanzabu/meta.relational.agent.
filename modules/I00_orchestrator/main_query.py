#!/usr/bin/env python3
import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from G00_retriever import Retriever, NoResultsFound


def run_query(query_text: str) -> int:
    """
    Execute a query and print the results.
    Returns 0 on success, 1 on error.
    """
    try:
        context_text, results = Retriever.retriever(query_text)
        Retriever.print_results(results, context_text, query_text)
        return 0
    except NoResultsFound as e:
        print(f"❌ {e}")
        return 1


def main_query():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return run_query(args.query_text)


if __name__ == "__main__":
    exit(main_query())
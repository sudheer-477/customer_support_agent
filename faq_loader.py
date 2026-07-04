"""
FAQ loader — parses care_intelligence_faq.md into structured Q&A records.

Keeping the FAQ content in its own markdown file (rather than hardcoded in
the pipeline script) means non-technical team members can update the FAQ
without touching any code — just edit the .md file and re-run indexing.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class FAQItem:
    id: int
    q: str
    a: str


def load_faq(path: str = "care_intelligence_faq.md") -> List[FAQItem]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Matches: **Q1: question text**\nanswer text (until next ** or end)
    pattern = r"\*\*Q(\d+):\s*(.+?)\*\*\s*\n(.+?)(?=\n\*\*Q\d+:|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    faq_items = []
    for num, question, answer in matches:
        faq_items.append(
            FAQItem(
                id=int(num),
                q=question.strip(),
                a=" ".join(answer.strip().split()),
            )
        )
    return faq_items


if __name__ == "__main__":
    items = load_faq("care_intelligence_faq.md")
    print(f"Loaded {len(items)} FAQ entries\n")
    for item in items:
        print(f"Q{item.id}: {item.q}")
        print(f"A{item.id}: {item.a}\n")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.ollama_client import OllamaClient


def build_table_prompt(text: str) -> str:
    return f"""
You are an expert at converting messy notes into clear study tables.

Task:
Analyze the extracted text and infer a meaningful table that helps a student
understand the content better.

Reasoning rules (INTERNAL ONLY):
- Identify concepts, categories, steps, or comparisons.
- Group related information logically.
- Create column names that reflect the inferred structure.

Output rules (STRICT):
- Output ONLY valid JSON.
- Do NOT include explanations or reasoning.
- Do NOT invent factual content beyond what is implied.
- All rows must match the number of columns.
- If a value is unclear, use "".

Output format (EXACT):
{{
  "columns": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["value", "value", "value"]
  ]
}}

Extracted text:
{text}
""".strip()


def build_mindmap_prompt(text: str) -> str:
    return f"""
You are an expert note distiller.

Task:
Convert the extracted text into a meaningful mind map that improves clarity
and understanding.

Reasoning rules (INTERNAL ONLY):
- Identify the central theme.
- Group related ideas under logical headings.
- Preserve relationships such as cause–effect, steps, or hierarchies.

Output rules (STRICT):
- Output ONLY a Markdown bullet hierarchy.
- Use '-' for bullets.
- Exactly ONE top-level root node.
- Maximum depth: 4 levels.
- No explanations, no summaries, no extra text.
- Do NOT add information not present or strongly implied.

Example format:
- Main Topic
  - Subtopic
    - Detail

Extracted text:
{text}
""".strip()


async def build_table(text: str, ollama_client: "OllamaClient", model: str) -> str:
    """Convert text to table format using LLM."""
    prompt = build_table_prompt(text)
    result = await ollama_client.generate(
        model=model,
        prompt=prompt,
        images=None,
    )
    return result


async def build_mindmap(text: str, ollama_client: "OllamaClient", model: str) -> str:
    """Convert text to mind map format using LLM."""
    prompt = build_mindmap_prompt(text)
    result = await ollama_client.generate(
        model=model,
        prompt=prompt,
        images=None,
    )
    return result


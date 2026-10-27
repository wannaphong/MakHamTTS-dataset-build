"""
Extract and clean high-quality Thai sentences from pythainlp/thai-wiki-dataset-v4.
Filters for pure Thai text suitable for TTS voice synthesis (30-120 chars, no maiyamok, no digits).
"""

import re
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThaiWikiExtractor")

THAI_PATTERN = re.compile(r"^[\u0e00-\u0e7f\s]+$")
THAI_DIGITS = set("๐๑๒๓๔๕๖๗๘๙0123456789")

def clean_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned

def is_valid_thai_tts_sentence(text: str, min_chars: int = 30, max_chars: int = 120) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if "ๆ" in text:
        return False
    if any(ch in THAI_DIGITS for ch in text):
        return False
    if not THAI_PATTERN.match(text):
        return False
    return True

def extract_wiki_sentences(
    target_count: int = 70000,
    output_file: str = "/workspace/labtts/som-tts-dataset-v1/text/thai_wiki.txt",
    min_chars: int = 30,
    max_chars: int = 120
):
    logger.info(f"Connecting to pythainlp/thai-wiki-dataset-v4 (target: {target_count} sentences)...")
    ds = load_dataset("pythainlp/thai-wiki-dataset-v4", split="train", streaming=True)

    seen = set()
    collected = []
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=target_count, desc="Extracting Thai Wiki Sentences")

    # Delimiters to split paragraphs into sentence clauses
    split_regex = re.compile(r"[\n\r\t\(\)\[\]\{\}\<\>\"\'«»“”‘’—–\-_/\\:;\.,!?]+")

    for item in ds:
        article_text = item.get("text", "")
        if not article_text:
            continue

        raw_chunks = split_regex.split(article_text)
        for chunk in raw_chunks:
            cleaned = clean_sentence(chunk)
            if is_valid_thai_tts_sentence(cleaned, min_chars=min_chars, max_chars=max_chars):
                if cleaned not in seen:
                    seen.add(cleaned)
                    collected.append(cleaned)
                    pbar.update(1)
                    if len(collected) >= target_count:
                        break

        if len(collected) >= target_count:
            break

    pbar.close()

    logger.info(f"Saving {len(collected)} sentences to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for s in collected:
            f.write(s + "\n")

    logger.info(f"Done! Saved {len(collected)} sentences to {output_file}")
    return len(collected)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Thai Wikipedia sentences for TTS")
    parser.add_argument("--count", type=int, default=70000, help="Number of sentences to extract")
    parser.add_argument("--output", type=str, default="/workspace/labtts/som-tts-dataset-v1/text/thai_wiki.txt", help="Output text file")
    parser.add_argument("--min-chars", type=int, default=30, help="Minimum character length")
    parser.add_argument("--max-chars", type=int, default=120, help="Maximum character length")
    args = parser.parse_args()

    extract_wiki_sentences(
        target_count=args.count,
        output_file=args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars
    )

"""Build bounded evidence and validate source identifiers without API calls."""
import math
import re
from collections import Counter

STOP_WORDS = set("вы мы и а но или в на с со по из для это как что ли о об у к да нет проводите можете расскажи пожалуйста".split())


def terms(text):
    return [word for word in re.findall(r"\w+", text.lower().replace("ё", "е"))
            if len(word) > 2 and word not in STOP_WORDS]


def passages(text, size=1000, overlap=200):
    # Overlap keeps headings and answers together, including in old indexes.
    for start in range(0, len(text), size - overlap):
        yield text[start:start + size]


def build_evidence(query, metadata, semantic_results, max_length=3000, top_k=3):
    query_terms = set(terms(query))
    candidates = []
    seen = set()
    for item in metadata:
        source = item['source']
        for passage in passages(item['text']):
            key = (source, passage)
            if key not in seen:
                seen.add(key)
                candidates.append((passage, source))
    if not candidates:
        return '', []
    counts = [Counter(terms(text)) for text, _ in candidates]
    frequencies = Counter(word for count in counts for word in count)
    semantic_rank = {source: rank for rank, (_, source, _) in
                     reversed(list(enumerate(semantic_results, 1)))}
    scored = []
    for (text, source), count in zip(candidates, counts):
        lexical = sum(math.log(1 + len(candidates) / frequencies[word]) *
                      count[word] / (count[word] + 1.2)
                      for word in query_terms if count[word])
        coverage = len(query_terms & count.keys()) / max(1, len(query_terms))
        score = lexical + coverage * 4 + 1 / (60 + semantic_rank.get(source, 10000))
        scored.append((score, text, source))
    scored.sort(key=lambda item: item[0], reverse=True)
    context, sources = '', []
    for _, text, source in scored:
        if len(sources) >= top_k:
            break
        # Do not fill the context with duplicate, overlapping passages.
        if any(text[:200] in previous for previous in context.split('\n\n')):
            continue
        filename = re.sub(r" \(часть \d+/\d+\)$", '', source)
        block = f"[{len(sources) + 1}] Файл: {filename}\n{text}\n\n"
        if len(context) + len(block) > max_length:
            continue
        context += block
        sources.append(filename)
    return context, sources


def cited_sources(answer, sources):
    """Only return sources explicitly cited by the model, from supplied evidence."""
    ids = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    if any(value < 1 or value > len(sources) for value in ids):
        return "Не удалось подтвердить ответ документами базы знаний. Уточните вопрос.", []
    used = list(dict.fromkeys(sources[value - 1] for value in ids))
    if not used:
        return "В найденных фрагментах базы знаний нет подтверждённого ответа. Уточните вопрос.", []
    answer = re.sub(r"[ \t]*\[\d+\]", "", answer).strip()
    return answer, used
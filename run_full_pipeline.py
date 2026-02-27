import os
import sys
import json
import re
import glob
import numpy as np
import torch
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple


ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(ROOT, "assignment_2_baseline_data", "baseline_data")
CLEANED_DIR = os.path.join(ROOT, "data", "docs")
QUERIES_PATH = os.path.join(ROOT, "leaderboard_queries.json")
CHUNKS_PATH = os.path.join(ROOT, "data", "chunks_v2.jsonl")
FAISS_DIR = os.path.join(ROOT, "data", "faiss_v2")
SUBMISSION_PATH = os.path.join(ROOT, "submission_Pidgeot_extractive.json")
SYSTEM_OUT_DIR = os.path.join(ROOT, "system_outputs")

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
READER_MODEL_NAME = "google/flan-t5-large"

CHUNK_SIZE = 200

HARDCODED_ANSWERS_BY_QUESTION = {}


def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'data-[a-z-]+="[^"]*"', '', text)
    text = re.sub(r'(loading|decoding|width|height|class|id|style|src|alt)="[^"]*"', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'[│├└─┌┐┘┬┤►▼▲◄○●◆■□★☆→←↑↓]', '', text)
    text = re.sub(r'(?:var|function|document|window|addEventListener)\s*[\({].*?[\)}];?', '', text, flags=re.DOTALL)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u00e2\u0080\u009c', '"').replace('\u00e2\u0080\u009d', '"')
    text = text.replace('\u00e2\u0080\u0099', "'")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    return text.strip()


def parse_wikipedia_html(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        content = soup.find("div", {"id": "bodyContent"})
    if not content:
        content = soup.find("div", {"class": "mw-parser-output"})
    if not content:
        content = soup

    for tag in content.find_all(["sup", "span"], class_=lambda x: x and ("reference" in str(x).lower() or "mw-editsection" in str(x).lower())):
        tag.decompose()
    for tag in content.find_all("div", class_=lambda x: x and ("navbox" in str(x).lower() or "sidebar" in str(x).lower() or "metadata" in str(x).lower())):
        tag.decompose()
    for tag in content.find_all("table", class_=lambda x: x and ("navbox" in str(x).lower() or "sidebar" in str(x).lower())):
        tag.decompose()

    parts = []
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text().replace(" - Wikipedia", "").strip()
        parts.append(title)

    for elem in content.find_all(["p", "li", "h1", "h2", "h3", "h4", "dd", "dt", "th", "td", "caption"]):
        text = elem.get_text(separator=" ", strip=True)
        text = re.sub(r'\[\s*\d+\s*\]', '', text)
        text = re.sub(r'\[\s*edit\s*\]', '', text)
        text = re.sub(r'\[\s*citation needed\s*\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and len(text) > 10:
            parts.append(text)

    result = "\n".join(parts)
    return clean_text(result)


def parse_text_doc(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = text.split('\n')
    cleaned = []
    skip_patterns = [
        r'^Skip to', r'^Toggle .* subsection', r'^Main menu',
        r'^From Wikipedia', r'^\[\s*edit\s*\]$', r'^Navigation menu',
        r'^Retrieved from', r'^Categories\s*:', r'^Hidden categories',
        r'^This page was last', r'^Content is available under',
        r'^opens in new tab', r'^Quick Links', r'^Search$',
        r'^Close$', r'^Menu$', r'^Sign [Ii]n', r'^Log [Ii]n',
        r'^Contact [Uu]s$', r'^Follow [Uu]s$', r'^Share$',
        r'^Back to top$', r'^Cookie', r'^Privacy Policy',
        r'^Terms of [Uu]se', r'^All [Rr]ights [Rr]eserved',
        r'^img\s+', r'^Block-image',
    ]
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append('')
            continue
        skip = False
        for pat in skip_patterns:
            if re.match(pat, line, re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned.append(line)

    result = '\n'.join(cleaned)
    return clean_text(result)


def load_all_documents() -> List[Dict]:
    documents = []
    
    print("Parsing baseline Wikipedia HTML files...")
    html_files = glob.glob(os.path.join(BASELINE_DIR, "*.htm"))
    for fpath in sorted(html_files):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        text = parse_wikipedia_html(fpath)
        if text and len(text) > 50:
            documents.append({
                "doc_id": f"wiki_{basename}",
                "text": text,
                "source": fpath,
                "type": "wikipedia"
            })
    print(f"  Loaded {len(documents)} Wikipedia documents")
    
    print("Loading scraped text documents...")
    txt_files = glob.glob(os.path.join(CLEANED_DIR, "*.txt"))
    scraped_count = 0
    skipped_count = 0
    
    noise_patterns = [
        r'pittsburghsymphony\.org',
        r'www\.pittsburghpa\.gov',
        r'events\.cmu\.edu',
        r'ppgpaintsarena\.com',
        r'www\.cmu\.edu_hr', r'www\.cmu\.edu_finance',
        r'www\.cmu\.edu_procurement', r'www\.cmu\.edu_alert',
        r'www\.cmu\.edu_brand', r'www\.cmu\.edu_advancement',
        r'www\.cmu\.edu_alumni', r'www\.cmu\.edu_legal',
        r'www\.cmu\.edu_audit', r'www\.cmu\.edu_police',
        r'www\.cmu\.edu_parking', r'www\.cmu\.edu_printing',
        r'www\.cmu\.edu_health-services', r'www\.cmu\.edu_housing',
        r'www\.cmu\.edu_disability', r'www\.cmu\.edu_title-ix',
        r'www\.cmu\.edu_compliance', r'www\.cmu\.edu_benefits',
        r'www\.cmu\.edu_payroll', r'www\.cmu\.edu_CMWA',
        r'www\.cmu\.edu_SETChange',
        r'www\.cmu\.edu_admission_admission_(?:apply|transfer|international|standardized|counselor|faq|where-am|consideration|support)',
        r'www\.cmu\.edu_admission_contact',
        r'www\.cmu\.edu_admission_costs-aid',
    ]
    
    for fpath in sorted(txt_files):
        basename = os.path.splitext(os.path.basename(fpath))[0].lower()
        
        is_noise = False
        for pat in noise_patterns:
            if re.search(pat, basename, re.IGNORECASE):
                is_noise = True
                break
        if is_noise:
            skipped_count += 1
            continue
        
        text = parse_text_doc(fpath)
        if text and len(text) > 50:
            documents.append({
                "doc_id": os.path.splitext(os.path.basename(fpath))[0],
                "text": text,
                "source": fpath,
                "type": "scraped"
            })
            scraped_count += 1
    print(f"  Loaded {scraped_count} scraped documents (skipped {skipped_count} noise)")
    print(f"  Total documents: {len(documents)}")
    
    return documents


def sentence_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    
    chunks = []
    current_words = []
    current_len = 0
    
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        
        if current_len + len(words) > chunk_size and current_words:
            chunks.append(' '.join(current_words))
            if overlap > 0:
                tail = current_words[-overlap:] if len(current_words) > overlap else current_words
                current_words = tail + words
                current_len = len(current_words)
            else:
                current_words = words
                current_len = len(words)
        else:
            current_words.extend(words)
            current_len += len(words)
    
    if current_words:
        chunks.append(' '.join(current_words))
    
    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    print("Building chunks...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc["text"])
        for i, chunk_text in enumerate(doc_chunks):
            cleaned = clean_text(chunk_text)
            if not cleaned or len(cleaned.split()) < 10:
                continue
            all_chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}__{i}",
                "text": cleaned,
                "source": doc["source"],
                "type": doc.get("type", "unknown")
            })
    
    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: List[Dict], path: str = CHUNKS_PATH):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved {len(chunks)} chunks to {path}")


def load_chunks(path: str = CHUNKS_PATH) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_faiss_index(chunks: List[Dict], model_name: str = EMBED_MODEL_NAME):
    import faiss
    from sentence_transformers import SentenceTransformer
    
    print(f"Building FAISS index with {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = [c["text"] for c in chunks]
    
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if (i // batch_size) % 20 == 0:
            print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    
    embeddings = np.vstack(all_embeddings).astype("float32")
    
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    
    return index, model


def tokenize_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'between', 'out', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where',
                  'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                  'own', 'same', 'so', 'than', 'too', 'very', 'and', 'but',
                  'or', 'if', 'while', 'that', 'this', 'these', 'those', 'it',
                  'its', 'also', 'just', 'about', 'up'}
    return [t for t in tokens if len(t) > 1 and t not in stop_words]


def build_bm25_index(chunks: List[Dict]):
    from rank_bm25 import BM25Okapi
    
    print("Building BM25 index...")
    corpus = [tokenize_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"  BM25 built over {len(corpus)} documents")
    return bm25


def retrieve_dense(index, model, query: str, topk: int = 100) -> List[Tuple[int, float]]:
    import faiss
    qv = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, topk)
    results = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
    return results


def retrieve_sparse(bm25, query: str, topk: int = 100) -> List[Tuple[int, float]]:
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[-topk:][::-1]
    results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
    return results


def rrf_merge(results_list: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    scores = {}
    for results in results_list:
        for rank, (idx, _) in enumerate(results, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(index, embed_model, bm25, query: str, topk: int = RETRIEVAL_TOPK) -> List[int]:
    dense_results = retrieve_dense(index, embed_model, query, topk=topk)
    sparse_results = retrieve_sparse(bm25, query, topk=topk)
    fused = rrf_merge([dense_results, sparse_results], k=60)
    return [idx for idx, _ in fused[:topk]]


def rerank_chunks(query: str, candidate_idxs: List[int], chunks: List[Dict], topk: int = RERANK_TOPK) -> List[Dict]:
    from sentence_transformers import CrossEncoder
    
    global _reranker
    if '_reranker' not in globals() or _reranker is None:
        print("  Loading reranker...")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    
    pairs = []
    valid_idxs = []
    for idx in candidate_idxs:
        if 0 <= idx < len(chunks):
            pairs.append((query, chunks[idx]["text"]))
            valid_idxs.append(idx)
    
    if not pairs:
        return []
    
    scores = _reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(scores)[::-1]
    
    results = []
    for i in order[:topk]:
        idx = valid_idxs[int(i)]
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "score": float(scores[int(i)]),
            "doc_id": chunks[idx]["doc_id"]
        })
    return results


_reranker = None

def load_generator(model_name: str = READER_MODEL_NAME):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    print(f"Loading generator: {model_name}...")
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return tokenizer, model, device


def clean_answer(answer: str) -> str:
    answer = answer.replace('\ufeff', ' ').replace('\ufffe', ' ')
    answer = answer.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    answer = re.sub(r'<[^>]+>', '', answer)
    answer = re.sub(r'data-[a-z-]+="[^"]*"', '', answer)
    answer = re.sub(r'(loading|decoding|width|height|class|id|style|src|alt)="[^"]*"', '', answer)
    answer = re.sub(r'https?://\S+', '', answer)
    answer = re.sub(r'Block-image[^"]*', '', answer)
    answer = re.sub(r'img\s+', '', answer)
    answer = re.sub(r'WikiMiniAtlas\s*', '', answer)
    answer = re.sub(r"\d+[°┬░]\d+[′'ΓÇ▓]\d+[\.\d]*[″\"ΓÇ│][NSEW]?", '', answer)
    answer = re.sub(r'[\-]?\d{1,3}\.\d{3,};\s*[\-]?\d{1,3}\.\d{3,}', '', answer)
    answer = re.sub(r'[\d.]+°[NSEW]', '', answer)
    answer = re.sub(r'\bCoordinates\b', '', answer)
    answer = re.sub(r'∩╗┐\s*/?', '', answer)
    answer = re.sub(r'┬░', '', answer)
    answer = re.sub(r'ΓÇ[▓│╖]', '', answer)
    answer = re.sub(r'\(ID\d+\)', '', answer)
    answer = re.sub(r'Added to NRHP\s+\w+\s+\d+,\s+\d+', '', answer)
    answer = re.sub(r'Designated PHLF', '', answer)
    answer = re.sub(r'Contributing property', '', answer)
    answer = re.sub(r'Pittsburgh Landmark - PHLF', '', answer)
    answer = re.sub(r'Historic district\b', '', answer)
    answer = re.sub(r'Significant dates', '', answer)
    answer = re.sub(r'\b(Home|about|vendors|map\s*&\s*Agenda|FAQS|buy tickets)\b', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'Vendors\s*\|', '', answer)
    answer = re.sub(r'\b\d{4}\s+(Taco|Food)\s+Festival\b', 'Taco Festival', answer)
    answer = re.sub(r'Share on (Facebook|Twitter|LinkedIn)\s*', '', answer)
    answer = re.sub(r'Email this to a friend\s*', '', answer)
    answer = answer.replace('\u201c', '"').replace('\u201d', '"')
    answer = answer.replace('\u2019', "'").replace('\u2018', "'")
    answer = answer.replace('\u2013', '-').replace('\u2014', '-')
    answer = answer.replace('ΓÇô', '-').replace("ΓÇÖ", "'")
    answer = answer.replace('ΓÇ£', '"').replace('ΓÇ¥', '"')
    answer = answer.replace('├⌐', 'é').replace('├¿', 'è')
    answer = answer.replace('├ó', 'â').replace('├¡', 'í')
    answer = answer.replace('├╝', 'ü').replace('├Â', 'ö')
    answer = answer.replace('Â®', '').replace('Â©', '').replace('â¢', '').replace('Â»', '')
    answer = re.sub(r'\|\s*\w+\s+\d{1,2}(st|nd|rd|th)?,?\s*\d{4}', '', answer)
    answer = re.sub(r'\s*\|\s*', ' ', answer)
    answer = re.sub(r'\s*/\s*/\s*', ' ', answer)
    answer = re.sub(r'\s+/\s+', ' ', answer)
    title_prefixes = [
        'Carnegie Mellon University Additional Facts and Details',
        'Pittsburgh History, Landmarks, and General Facts',
        'Pittsburgh Food and Restaurant Guide',
        'Pittsburgh Neighborhoods and Shopping',
    ]
    for prefix in title_prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    answer = ' '.join(answer.split()).strip()
    answer = answer.strip('.,;:- /')
    return answer


def extract_best_sentences(question: str, contexts: List[Dict], max_sents: int = 3) -> str:
    q_lower = question.lower()
    q_words = set(re.findall(r'\b\w{3,}\b', q_lower))
    q_stop = {'what', 'which', 'where', 'when', 'who', 'how', 'does', 'did',
              'the', 'and', 'for', 'are', 'was', 'has', 'have', 'been',
              'its', 'that', 'this', 'with', 'from', 'about', 'known',
              'carnegie', 'mellon', 'university', 'pittsburgh', 'cmu'}
    q_content_words = q_words - q_stop
    
    scored_sents = []
    seen = set()
    for c in contexts[:5]:
        text = c["text"] if isinstance(c, dict) else c
        sents = re.split(r'(?<=[.!?])\s+', text)
        for sent in sents:
            sent = sent.strip()
            if len(sent) < 20 or sent in seen:
                continue
            seen.add(sent)
            s_lower = sent.lower()
            s_words = set(re.findall(r'\b\w{3,}\b', s_lower))
            overlap = len(q_content_words & s_words)
            new_info = len(s_words - q_words)
            len_penalty = 1.0 if len(sent) < 300 else 0.5
            score = (overlap * 2 + new_info * 0.5) * len_penalty
            scored_sents.append((sent, score))
    
    scored_sents.sort(key=lambda x: x[1], reverse=True)
    best = [s[0] for s in scored_sents[:max_sents]]
    return ' '.join(best)



def compress_answer_minimal(tokenizer, model, device, question: str, answer: str, contexts: List[Dict], max_new_tokens: int = 48) -> str:
    context_texts = [c.get('text', '') for c in contexts] if contexts else []
    joined = ' '.join(context_texts)
    if len(joined) > 2500:
        joined = joined[:2500]

    prompt = (
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Supporting context: {joined}\n"
        "Task: Produce a single concise sentence that directly answers the question using evidence from the supporting context. "
        "Do NOT include menus, navigation, or unrelated boilerplate. If you cannot produce a concise answer, reply with 'NO_ANSWER'.\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    compressed = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    compressed = clean_answer(compressed)
    if not compressed:
        return ""
    if compressed.lower().startswith('no_answer') or compressed.lower().startswith('no answer'):
        return ""
    if compressed and compressed[-1] not in '.!?':
        compressed = compressed + '.'
    return compressed


def generate_answer(tokenizer, model, device, question: str, contexts: List[Dict], max_new_tokens: int = GENERATION_MAX_TOKENS) -> str:
    context_texts = []
    for c in contexts[:5]:
        text = c["text"] if isinstance(c, dict) else c
        context_texts.append(text)
    
    joined_context = "\n\n".join(context_texts)
    
    max_context_chars = 2800
    if len(joined_context) > max_context_chars:
        joined_context = joined_context[:max_context_chars]
    
    prompt = (
        f"Read the context and answer the question. Include all relevant details "
        f"from the context in your answer.\n\n"
        f"Context: {joined_context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    answer = clean_answer(answer)
    return answer


def main():
    import time
    start_time = time.time()
    
    query_path = sys.argv[1] if len(sys.argv) > 1 else QUERIES_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else SUBMISSION_PATH
    is_leaderboard = "leaderboard" in os.path.basename(query_path).lower()
    
    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {os.path.basename(query_path)}")
    
    faiss_path = os.path.join(FAISS_DIR, "index.faiss")
    
    if os.path.exists(CHUNKS_PATH):
        print("Loading cached chunks...")
        chunks = load_chunks()
        print(f"  Loaded {len(chunks)} cached chunks")
    else:
        print("Building chunks from all data sources...")
        documents = load_all_documents()
        chunks = build_chunks(documents)
        save_chunks(chunks)
    
    import faiss
    from sentence_transformers import SentenceTransformer
    
    if os.path.exists(faiss_path):
        print("Loading cached FAISS index...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        index = faiss.read_index(faiss_path)
        print(f"  FAISS index loaded with {index.ntotal} vectors")
    else:
        index, embed_model = build_faiss_index(chunks)
    
    bm25 = build_bm25_index(chunks)
    tokenizer, gen_model, device = load_generator()
    
    print(f"\nProcessing {len(queries)} queries...")
    results = {}
    if is_leaderboard:
        results["andrewid"] = "t2"
    
    for i, item in enumerate(queries):
        qid = str(item.get("id", i + 1))
        question = item.get("question", item.get("query", ""))
        
        candidate_idxs = hybrid_retrieve(index, embed_model, bm25, question)
        reranked = rerank_chunks(question, candidate_idxs, chunks, topk=RERANK_TOPK)
        
        if reranked:
            gen_answer = generate_answer(tokenizer, gen_model, device, question, reranked)
        else:
            gen_answer = ""
        
        if not gen_answer or len(gen_answer.strip()) < 2:
            answer = extract_best_sentences(question, reranked, max_sents=2)
            answer = clean_answer(answer)
        else:
            answer = clean_answer(gen_answer)
        
        if len(answer) > 400:
            sents = re.split(r'(?<=[.!?])\s+', answer)
            trimmed = []
            total = 0
            for s in sents:
                if total + len(s) > 350 and trimmed:
                    break
                trimmed.append(s)
                total += len(s) + 1
            answer = ' '.join(trimmed)
        
        if answer and answer[-1] not in '.!?':
            answer = answer.rstrip('.,;:- /') + '.'
        
        results[qid] = answer
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(queries)}] Q: {question[:55]}...")
            print(f"           A: {answer[:90]}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved submission to {output_path}")
    
    Path(SYSTEM_OUT_DIR).mkdir(parents=True, exist_ok=True)
    sys_out_name = "leaderboard_output.json" if is_leaderboard else "system_output_1.json"
    sys_out_path = os.path.join(SYSTEM_OUT_DIR, sys_out_name)
    with open(sys_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done!")


if __name__ == "__main__":
    main()

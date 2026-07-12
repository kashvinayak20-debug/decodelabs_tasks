"""
PROJECT 3: AI RECOMMENDATION LOGIC
"""

import csv
import math
from collections import Counter



# STEP 1: INGESTION - Load the dataset

def load_dataset(filepath):
    """Reads the CSV and returns a dict: {role_name: [skill1, skill2, ...]}"""
    roles = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skills = [s.strip().lower() for s in row["skills"].split(";")]
            roles[row["role"]] = skills
    return roles



# STEP 2: VECTOR MAPPING - Build the shared vocabulary

def build_vocabulary(roles):
    """Every unique skill across all roles becomes one dimension
    in our vector space (the 'shared vocabulary' from the slides)."""
    vocab = set()
    for skills in roles.values():
        vocab.update(skills)
    return sorted(vocab)



# STEP 3: TF-IDF - The math from "Upgrading Feature Extraction"

def compute_tf(skills_list, vocab):
    """Term Frequency: how much each skill matters WITHIN this one document.
    TF = (count of term in doc) / (total terms in doc)"""
    counts = Counter(skills_list)
    total_terms = len(skills_list)
    return {term: counts.get(term, 0) / total_terms for term in vocab}


def compute_idf(all_documents, vocab):
    """Inverse Document Frequency: penalizes skills that appear
    in almost every role (generic) and rewards rare, specific ones.
    IDF = log(Total Documents / Documents containing the term)"""
    n_docs = len(all_documents)
    idf = {}
    for term in vocab:
        doc_count = sum(1 for doc in all_documents if term in doc)
        # +1 smoothing avoids division by zero for unseen terms
        idf[term] = math.log(n_docs / (doc_count + 1)) + 1
    return idf


def build_tfidf_vector(skills_list, vocab, idf):
    """Combines TF and IDF: weight = TF(term) * IDF(term)"""
    tf = compute_tf(skills_list, vocab)
    return {term: tf[term] * idf[term] for term in vocab}



# STEP 4: THE SIMILARITY ENGINE - Cosine Similarity

def cosine_similarity(vec_a, vec_b, vocab):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)
    Measures the ANGLE between two vectors -> ignores raw magnitude,
    focuses purely on orientation/direction of preference.
    """
    dot_product = sum(vec_a[t] * vec_b[t] for t in vocab)
    magnitude_a = math.sqrt(sum(vec_a[t] ** 2 for t in vocab))
    magnitude_b = math.sqrt(sum(vec_b[t] ** 2 for t in vocab))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold Start Problem: a zero vector can't be compared
    return dot_product / (magnitude_a * magnitude_b)


# STEP 5: THE 4-STEP RANKING PIPELINE
# (Ingestion -> Scoring -> Sorting -> Filtering)

def recommend(user_skills, roles, vocab, idf, top_n=3):
    # --- Ingestion ---
    user_skills = [s.strip().lower() for s in user_skills]
    user_vector = build_tfidf_vector(user_skills, vocab, idf)

    # --- Scoring ---
    scores = []
    for role_name, skills in roles.items():
        role_vector = build_tfidf_vector(skills, vocab, idf)
        score = cosine_similarity(user_vector, role_vector, vocab)
        scores.append((role_name, score))

    # --- Sorting ---
    scores.sort(key=lambda x: x[1], reverse=True)

    # --- Filtering (Top-N) ---
    return scores[:top_n]



# MAIN PROGRAM - Handles the Cold Start Problem via a survey-style
# input loop, then runs the full IPO pipeline.

def main():
    
    print(" TECH STACK RECOMMENDER ")
    

    roles = load_dataset("raw_skills.csv")
    vocab = build_vocabulary(roles)
    idf = compute_idf(list(roles.values()), vocab)

    print("\nTell us at least 3 of your skills (comma-separated).")
    print("Example: Python, Cloud Computing, Automation\n")

    raw = input("Your skills: ").strip()
    user_skills = [s.strip() for s in raw.split(",") if s.strip()]

    # Bypass Cold Start: if fewer than 3 skills given, fall back to a
    # trending/default skill set instead of returning a zero-vector match.
    if len(user_skills) < 3:
        print("\n[Notice] Fewer than 3 skills detected — using a trending")
        print("baseline (Python, Automation, SQL) to avoid a Cold Start.")
        user_skills = list(set(user_skills + ["python", "automation", "sql"]))

    results = recommend(user_skills, roles, vocab, idf, top_n=3)

    print("\n--- TOP 3 RECOMMENDED CAREER PATHS ---")
    for rank, (role, score) in enumerate(results, start=1):
        match_pct = round(score * 100, 1)
        print(f"{rank}. {role:<25} | Match Score: {match_pct}%")

    print("\nDone. This ranking is based purely on angular alignment")
    print("(Cosine Similarity) between your skills and each role's profile.")


if __name__ == "__main__":
    main()

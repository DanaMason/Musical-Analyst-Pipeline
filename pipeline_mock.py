def load_music_csv(path=None):
    print("[BACKEND] load_music_csv() called - working")
    csv_text = "track_title,genre_tags\nTest Track A,test\nTest Track B,test\n"
    lookup = {"Test Track A": ("Test Artist", "Test Album", 1),
              "Test Track B": ("Test Artist", "Test Album", 2)}
    context_lookup = {"Test Track A": {"genre_tags": "test"},
                      "Test Track B": {"genre_tags": "test"}}
    return csv_text, lookup, context_lookup


def load_phi4():
    print("[BACKEND] load_phi4() called - working")
    return "mock-phi4-model", "mock-phi4-tokenizer"


def load_mert():
    print("[BACKEND] load_mert() called - working")
    return "mock-mert-model", "mock-mert-processor"


def similarity_llm(query, csv_text, model, tok):
    print(f"[BACKEND] similarity_llm() called - working (query: {query!r})")
    return [("Test Track A", 95.00), ("Test Track B", 30.00)]


def query_mert(ranked, lookup, context_lookup, mert, processor):
    print(f"[BACKEND] query_mert() called - working ({len(ranked)} ranked titles in)")
    return [{"filename": f"{i:02d} {title}.wav",
             "title": title,
             "clap_score": score,
             "mert_embedding": [0.0],
             "features": {"tempo_bpm": 120.0, "brightness_hz": 1500.0,
                          "energy_rms": 0.1000, "percussiveness_zcr": 0.1000},
             "context": context_lookup.get(title, {})}
            for i, (title, score) in enumerate(ranked, 1)]


def mert_pairwise(mert_results):
    print("[BACKEND] mert_pairwise() called - working")
    return "  Test Track A  vs  Test Track B:  1.000"


def query_output_llm(user_query, mert_results, csv_text, model, tok):
    print("[BACKEND] query_output_llm() called - working")
    return "The system is working!"
import os
import argparse
import re
import json
import difflib
import unicodedata
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Wav2Vec2FeatureExtractor, AutoModel # , ClapModel, ClapProcessor,	# Legacy Code
import librosa
import torch
import numpy as np
import pandas as pd
from json_repair import repair_json
from transformers import BitsAndBytesConfig

SDSO_ROOT = os.environ.get("SDSO_ROOT", "/home/ren-admin/sdso")
CSV_PATH = "/home/ren-admin/sdso/Musical-Analyst-Pipeline/new_metadata.csv"
AUDIO_ROOT = "/home/ren-admin/sdso/audio_data" 
PROMPT_COLS = ["track_title", "genre_tags", "english_translation",
               "song_description", "liner_notes", "featured_performers"]

os.environ["HF_HOME"] = "/home/ren-admin/sdso/hf_cache"  # Set Hugging Face cache directory

PHI4_ID = os.environ.get("PHI4_ID", "microsoft/phi-4")
MERT_ID = os.environ.get("MERT_ID", "m-a-p/MERT-v1-95M")

USE_4BIT = os.environ.get("SDSO_4BIT", "1") == "1"
CTX_LIMIT = 16384
MERT_SECONDS = float(os.environ.get("MERT_SECONDS", "60"))

# Prompt Engineering Below:

# Original
# input_role = """
# You convert a conductor's question into ONE short CLAP search caption.
# Output ONLY the caption, nothing else: a single sentence (max 25 words) describing the SOUND
# — instrumentation, tempo, rhythm, vocal style, emotional tone.
# No preamble, no reasoning, no lists, no quotation marks.
# Example input: What pieces feel like a Lakota war song?
# Example output: A traditional Lakota war song with powerful drumming and driving, intense vocal chanting.
# """

input_role = """
You are a specialist comparing textual descriptions of Lakota and Native American music to audio recording liner notes, transcriptions, genre tags, and other metadata.
You will be given CSV data compiled with song titles alongside the corresponding audio file descriptions.

Being exact, this will include: genre_tags, english transcriptions of the songs, descriptions of the songs, liner notes from the songs, performers of the song, and the title of the song.
Some entries will be missing some of these fields. Use whatever information you have available to best respond to any given question.

Your end goal is to pair the conductor/composer's question with the 10 most relevant songs from the CSV data based on these features given to you.
Only use track titles that appear verbatim in the CSV's track_title column - never invent or paraphrase a title.

Return your ranked answer as a JSON array of exactly 10 items, each item a two-element array [track_title, similarity_score], ordered most to least relevant. Round each score to the nearest hundredth. 
Output only the JSON array - no preamble, no explanation, no markdown fences.

Example:
[
  ["Black Hills Olowan", 91.50],
  ["Lakota National Anthem (Flag Song) and Veterans' Song", 87.25],
  ["Little Big Horn Battle Song", 84.00],
  ["Victory Songs (Waktégli Olówaŋ) I. Introduction", 79.75],
  ...
]
"""

output_role = """
You are a cross-cultural musical analyst assistant, aiding a conductor compare Lakota and Western Classical music. You are given a conductor's question, objective 
acoustic measurements per piece (tempo, spectral brightness, RMS energy, percussiveness), and pairwise similarity scores from a learned musical-representation model
called MERT. This model gains a high understanding of musical similarities on an emotional level. Higher similarity scores among MERT embeddings represent more 
similar songs. I want you to take these features and the similarity scores, as well as the initial prompt from the user, and generate an ideal response to them.

You are additionally given the full metadata CSV for the corpus - the same data used to retrieve these pieces - containing track titles, genre tags, English
translations, song descriptions, liner notes, and featured performers for every track. Use this documented metadata to ground your cultural and emotional claims
in real context, but NEVER invent details, instrumentation, lyrics, or history that are not present in the metadata or the acoustic measurements.

Further, you must ensure you do NOT hallucinate false information about the Lakota culture or the songs themselves. Answers must be logically grounded in provided
reference data and materials, or extremely well known and widely accepted knowledge about classical musical inforamtion (e.g., the works of Bach, Beethoven, Mozart, and other well-known composers).

They will be conducting the piece for a later performance, so you will want everything you think of and generate to take into account the fact that the conductor is 
likely highly familiar with music terminology and classical music. However, they are likely unfamiliar with the emotional feeling and meaning behind various Lakota
signals in songs. They are likely going to want you to provide a deeper and more meaningful understanding of the underlying similar songs (related to their provided 
feelings in their initial prompt given). So, you want to assist a highly intellectual musical composer attempting to generate cross-cultural musical pieces. 

Furthermore, give them clear and useful advise with their level of knowledge in mind, and keeping in mind that they likely need to just further understand the differences
amongst the classical and Lakota cultures. They know music, they just need to understand the emotional feel of the Lakota songs to encorporate into new cross-cultural 
blended musical pieces. 

Treat the retrieved pieces as a set. Note the range each acoustic measurement spans across all of them before discussing individual pieces, 
rather than treating the top-ranked piece as the baseline the others are compared against.

Focus on enhancing an already general perspective on the type of music being asked about
such that the user can deepen understanding of the emotional feel of the pieces and their cross-cultural relevance. 
Particularly for writting music purposes, but if specifically asked about just understanding pieces or types of songs,
provide that knowledge instead of primarily focusing on writting music.

When writing your response, be clear, concise, and be grounded factually. Do not halucinate and give a fully in-depth understanding of the emotional feel of these pieces
provided. You want to relate these to their initial prompt as well as speciifc information useful for these professionals. You need to imply how these are 
cross-culturally relavenet. Ground all claims with provided numbers if possible. NEVER invent instrumentation or lyrics or songs or knowledge that was not given.

As a final note, if they ask you about features of classical music reference common sources of knowledge or pretraining data
that is well-known and widely accepted in classical music study, such as the works of Bach, Beethoven, Mozart, and other well-known composers. 
"""

CONTEXT_COLS = ["genre_tags", "song_description", "liner_notes", "english_translation", "featured_performers"]


def load_music_csv(path: str = CSV_PATH):
    df = pd.read_csv(path).fillna("")
    lookup = {row.track_title: (row.album_artist_or_ensemble, row.album_title, int(row.track_number))
              for row in df.itertuples()}
    context_lookup = {row.track_title: {c: getattr(row, c) for c in CONTEXT_COLS}
                       for row in df.itertuples()}
    csv_text = df[PROMPT_COLS].to_csv(index=False)
    return csv_text, lookup, context_lookup

# Load Phi 4.0 Model
def load_phi4():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92, 0)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,   
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(PHI4_ID)
    model = AutoModelForCausalLM.from_pretrained(
        PHI4_ID, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return model, tok

# Load MERT Model
def load_mert():
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
    mert = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True, device_map="auto")
    mert.eval()
    return mert, processor

# Receives the query from the user
def receive_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    # print(f"Query received: '{query}'")
    return query

# Sends the query to an LLM
def similarity_llm(query: str, csv_text: str, model, tok) -> list:
    user_msg = f"CSV data:\n{csv_text}\nConductor's question:\n{query}"

    messages = [
        {"role": "system", "content": input_role},
        {"role": "user", "content": user_msg}
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)

    response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM output: {response[:200]}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        parsed = repair_json(match.group(0), return_objects=True)

    ranked = [(title, float(score)) for title, score in parsed][:10]
    
    print(f"Similarity LLM ranked {len(ranked)} tracks:")
    for rank, (title, score) in enumerate(ranked, 1):
        print(f"  {rank:2d}. [{score:6.2f}]  {title}")

    del inputs, outputs
    torch.cuda.empty_cache()
    return ranked

# Remove weird characters and normalize strings for comparison
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Create an index of albums in the audio root directory for quick lookup
def build_album_index(root: str = AUDIO_ROOT) -> dict:
    index = {}
    for d in Path(root).rglob("*"):
        if any(part.startswith(".") or part == "__MACOSX" for part in d.relative_to(root).parts):
            continue
        if d.is_dir() and any(d.glob("*.wav")):
            index[normalize(d.name)] = d
    return index

# Resolve the album directory based on the album title and the index
def resolve_album_dir(album_title: str, index: dict):
    key = normalize(album_title)

    if key in index:
        return index[key]
    
    for folder_key, path in index.items():
        if key in folder_key or folder_key in key:
            return path
        
    close = difflib.get_close_matches(key, index.keys(), n=1, cutoff=0.5)
    return index[close[0]] if close else None

# The MERT model implementation
def query_mert(ranked: list, lookup: dict, context_lookup: dict, mert, processor) -> list:
    # Get embeddings per top result

    album_index = build_album_index()
    mert_results = []

    # Match filenames to audio for downloading and processing
    for title, score in ranked:
        if title not in lookup:
            close = difflib.get_close_matches(title, lookup.keys(), n=1, cutoff=0.8)
            if not close:
                print(f"  SKIP {title}: not in CSV lookup")
                continue
            title = close[0]
        
        artist, album, track_num = lookup[title]
        album_dir = resolve_album_dir(album, album_index)
        if album_dir is None:
            print(f"  SKIP {title}: no folder found for album '{album}'")
            continue
        
        matches = list(album_dir.glob(f"{track_num:02d} *"))

        if not matches:
            print(f"  SKIP {title}: no file starting with {track_num:02d} in {album_dir}")
            continue

        # Load audio, extract features, and get MERT embedding

        try:
            audio, _ = librosa.load(matches[0], sr=24000, mono=True, duration=60)               # Load in audio at 24kHz as a 1D array of audio samples (duration is not specified, so it will load the entire file, but only later use the first 60 seconds for feature extraction)
            inputs = processor(audio, sampling_rate=24000, return_tensors="pt", padding=True)   # Extract normalized/formatted features using MERT processor - similar to tokenizing
            inputs = {k: v.to(next(mert.parameters()).device) for k, v in inputs.items()}       # Formats inputs to the correct device (CPU or GPU) for the MERT model

            with torch.no_grad():
                out = mert(**inputs, output_hidden_states=True)     # Get MERT Embedding

            # Extract the mean of the last hidden layer into a single embedding, convert to a numpy array, and retrieve features using librosa in extract_musical_features
            embedding = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            del out, inputs
            torch.cuda.empty_cache()
            
            feats = extract_musical_features(matches[0])

            mert_results.append({"filename": str(matches[0]), "title": title, "clap_score": score,
                          "mert_embedding": embedding, "features": feats,
                          "context": context_lookup.get(title, {})})
            print(f"  OK  {title} -> {matches[0].name}")

        except Exception as e:
            print(f"  SKIP {title}: {e}")

    torch.cuda.empty_cache()
    return mert_results

# Extract musical features from audio example using librosa

# Tempo = BPM
# Brightness = Average Frequency (can give insight into things like pitch and timbre)
# RMS Energy = Average Energy (features like 'loudness')
# Percussiveness = Things like rythm (smoothness of music, distinct drum beats, etc.)

def extract_musical_features(path) -> dict:
    # Load audio sample in at 22.05kHz as a 1D array of audio samples, and limit to the first 60 seconds for feature extraction
    y, sr = librosa.load(path, sr=22050, mono=True, duration=60)        

    # The following lines extract musical features including tempo, spectral centroid (brightness), root mean square energy, and zero crossing rate (percussivenesS).
    # These are common in music analysis, and are all formatted as floats for easier later analysis and comparison.
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    tempo = float(np.atleast_1d(tempo)[0])                                     
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    return {"tempo_bpm": round(tempo, 1),
            "brightness_hz": round(centroid, 1),
            "energy_rms": round(rms, 4),
            "percussiveness_zcr": round(zcr, 4)}

# Computes pairwise similarity between all MERT embeddings and returns a formatted string of results

def mert_pairwise(mert_results: list) -> str:
    # Load all embeddings as numpy arrays, normalize, and compute cosine similary between all pairs.
    embs = np.array([r["mert_embedding"] for r in mert_results])                                        # Each song gets a row in this matrix, each column is a dimension of the embedding.
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sims = norm @ norm.T                          

    # Retrieve names from the accompanying file names.

    names = [Path(r["filename"]).name[:45] for r in mert_results]

    lines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lines.append(f"  {names[i]}  vs  {names[j]}:  {sims[i, j]:.3f}")

    return "\n".join(lines) if lines else "  (only one piece retrieved)"

# Utilize prior results to generate a logical final output in response to the user's initial query.

def query_output_llm(user_query: str, mert_results: list, csv_text: str, model, tok) -> str:

    # Generate natural langauge strings describing the features and pairwise similarity from the prior results, and format for final LLM input.

    feat_lines = []
    for res in mert_results:
        f, c = res["features"], res["context"]
        ctx = "; ".join(f"{k}: {v}" for k, v in c.items() if v)
        feat_lines.append(
            f"- {res['title']}: tempo {f['tempo_bpm']} BPM, brightness {f['brightness_hz']} Hz, "
            f"energy {f['energy_rms']}, percussiveness {f['percussiveness_zcr']}\n"
            f"  context: {ctx if ctx else '(none provided)'}"
        )
    feats_text = "\n".join(feat_lines)
    sims_text = mert_pairwise(mert_results)

    user_msg = (
        f"Conductor's question:\n{user_query}\n\n"
        f"Full corpus metadata (same CSV used for retrieval):\n{csv_text}\n\n"
        f"Retrieved pieces - acoustic measurements:\n{feats_text}\n\n"
        f"MERT pairwise similarity (1.0 = identical):\n{sims_text}"
    )

    # The rest is similar to the input LLM, but with a different prompt to guide output.

    messages = [{"role": "system", "content": output_role},
                {"role": "user", "content": user_msg}]
    
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=800, do_sample=False, pad_token_id=tok.eos_token_id)

    response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del inputs, outputs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return response

def main():
    parser = argparse.ArgumentParser(description="SDSO Music Analysis Pipeline")
    parser.add_argument("query", type=str, help="Natural language music query")
    args = parser.parse_args()

    query = receive_query(args.query)
    csv_text, lookup, context_lookup = load_music_csv()

    phi_model, phi_tok = load_phi4()
    mert, mert_proc = load_mert()

    top_results = similarity_llm(query, csv_text, phi_model, phi_tok)
    final_results = query_mert(top_results, lookup, context_lookup, mert, mert_proc)

    # Original:
    # clap_input = query_llm(query)
    # clap_results = query_clap(clap_input)
    # mert_results = query_mert(clap_results)
    
    print("\nMERT Results:")
    for res in final_results:
        f = res["features"]
        print(f"  {Path(res['filename']).name}")
        print(f"    tempo: {f['tempo_bpm']} BPM, "
              f"brightness: {f['brightness_hz']} Hz, energy: {f['energy_rms']}, "
              f"percussiveness: {f['percussiveness_zcr']}")
        
        # print(f"    CLAP similarity: {res['clap_score']:.4f}  |  tempo: {f['tempo_bpm']} BPM, "
        #       f"brightness: {f['brightness_hz']} Hz, energy: {f['energy_rms']}, "
        #       f"percussiveness: {f['percussiveness_zcr']}")
        
    if len(final_results) > 1:
        print("\nMERT Pairwise Similarity (1.0 = identical):")
        print(mert_pairwise(final_results))

    if final_results:
        print("\n=== Cross-Cultural Analysis ===")
        print(query_output_llm(query, final_results, csv_text, phi_model, phi_tok))

if __name__ == "__main__":
    main()





# Original CLAP implementation code
# # Sends the query to an LLM
# def query_llm(query: str) -> str:
#     snap = sorted(Path("/home/share/SDSO/hf_cache/hub").glob(
#         "models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*"     # Choses model type
#     ))[-1]

#     # Load LLM model and tokenizer
#     tok = AutoTokenizer.from_pretrained(snap)
#     model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float16, device_map="auto")
#     model.eval()

#     messages = [
#         {"role": "system", "content": input_role},
#         {"role": "user", "content": query}
#     ]

#     text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tok(text, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)

#     response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

#     quoted = re.findall(r'"([^"]{10,200})"', response)
#     lines = [l.strip() for l in response.splitlines() if l.strip()]
#     caption = (quoted[-1] if quoted else (lines[0] if lines else response)).strip('"')[:300]

#     print(f"First LLM Output: '{caption}'")
#     del model, tok, inputs, outputs
#     torch.cuda.empty_cache()
#     return caption

# # The CLAP model implemantation
# def query_clap(query: str, top_k: int = 10, mode: str = "text") -> list:
#     clap_snap = sorted(Path("/home/share/SDSO/hf_cache/hub").glob(
#         "models--laion--larger_clap_music/snapshots/*" # Load model from cache
#     ))[-1]

#     # Load CLAP model and processor
#     processor = ClapProcessor.from_pretrained(clap_snap)
#     model = ClapModel.from_pretrained(clap_snap)
#     model.eval()

#     # Encode the query using the CLAP processor
#     inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
    
#     # Version proof model output for consistency in CLAP embedding space

#     with torch.no_grad():
#         out = model.get_text_features(
#             input_ids=inputs["input_ids"].to(model.device),
#             attention_mask=inputs["attention_mask"].to(model.device))

#     text_vec = out.pooler_output if hasattr(out, "pooler_output") else out
#     text_embeddings = text_vec.squeeze().cpu().numpy()
    
#     # Load pre-computed embeddings for music
#     emb_file = {"text": "clap_text_embeddings.npy", "fused": "clap_fused_embeddings.npy", "audio": "clap_embeddings.npy"}[mode]
#     base = "/home/share/SDSO/cluster_results/clap"
#     audio_embs = np.load(f"{base}/{emb_file}", allow_pickle=True)
#     filenames = Path(f"{base}/clap_files.txt").read_text().strip().splitlines()

#     # Renormalize text embeddings if using text or fused mode
#     if mode in ("text", "fused"):
#         text_mean = np.load(f"{base}/clap_text_mean.npy")
#         text_embeddings = text_embeddings - text_mean

#     # Compute cosine similarity
#     audio_norm = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-9)
#     text_norm = text_embeddings / (np.linalg.norm(text_embeddings) + 1e-9)
#     similarities = np.dot(audio_norm, text_norm)

#     top_indices = np.argsort(similarities)[::-1][:top_k]
#     results = [(filenames[i], float(similarities[i])) for i in top_indices]

#     print(f"Top matches ({mode}): ")
#     for fname, score in results:
#         print(f"{fname}: {score:.4f}")
    
#     return results

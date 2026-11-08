"""Explicit frozen MiniLM encoding with exact token chunks and weighted pooling."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from .preparation import load, save, guarded

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = '7dbbc90392e2f80f3d3c277d6e90027e55de9125'
PINNED_FILES = {'tokenizer_config.json': 'acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b', 'special_tokens_map.json': '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3', 'config.json': '953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41', 'config_sentence_transformers.json': '061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3', 'tokenizer.json': 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037', 'sentence_bert_config.json': 'fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7', 'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3', 'pytorch_model.bin': 'c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256', 'data_config.json': '32edcb108fc2516b920734a862ae0692bcae1c5d45d5f8d972cb0d53434a4c54', 'modules.json': '84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf', '1_Pooling/config.json': '4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23'}

def split_tokens(ids, budget=254):
    if not ids or budget <= 0:
        raise ValueError("Nonempty content and positive chunk budget required")
    return [dict(start=start,stop=min(start+budget,len(ids)),ids=list(ids[start:start+budget]))
            for start in range(0,len(ids),budget)]


def pool_chunks(vectors, content_counts):
    values, weights = np.asarray(vectors,dtype=float), np.asarray(content_counts,dtype=float)
    if values.ndim!=2 or weights.shape!=(len(values),) or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("Finite chunk vectors and positive content-token weights required")
    norms = np.linalg.norm(values,axis=1)
    if np.any(norms<=0):
        raise ValueError("Zero chunk vector")
    mean = np.average(values/norms[:,None],axis=0,weights=weights)
    norm = np.linalg.norm(mean)
    if norm<=0:
        raise ValueError("Zero pooled document vector")
    return mean/norm



def snapshots_canonical_sha256(snapshots):
    """Fingerprint parsed snapshot content independently of JSON/gzip formatting."""
    return hashlib.sha256(json.dumps(snapshots, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_model(model_dir):
    model_dir = Path(model_dir)
    for name, expected in PINNED_FILES.items():
        if hashlib.sha256((model_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned encoder file differs: " + name)
    # Additional modules/weights cannot silently override the pinned model.
    if (model_dir/"model.safetensors").exists():
        raise ValueError("Unverified alternative weights; use the pinned pytorch_model.bin only")
    return model_dir

def encode(snapshots_path, model_dir, output, allow_encode=False):
    if not allow_encode:
        raise ValueError("Encoding requires explicit --allow-encode")
    output = guarded(output)
    if output.exists():
        raise ValueError("Use a fresh encoding output directory")
    model_dir = verify_model(model_dir)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    torch.manual_seed(718)
    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 256
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer.num_special_tokens_to_add(pair=False) != 2 or tokenizer.truncation_side != "right":
        raise ValueError("Expected BERT CLS/SEP with right truncation")
    snapshots = load(snapshots_path)
    descriptors, inputs = {}, {}
    for sid, record in sorted(snapshots.items()):
        if record["snapshot_id"] != sid or sid != f"{record['pageid']}:{record['revision_id']}":
            raise ValueError("Normalized snapshot identity mismatch")
        if record["missing"]:
            if record["text"]:
                raise ValueError("Missing snapshot has content")
            descriptors[sid] = []
            continue
        ids = tokenizer(record["text"], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        chunks = split_tokens(ids)
        native = tokenizer(record["text"], add_special_tokens=True, truncation=True, max_length=256, verbose=False)["input_ids"]
        if native != [tokenizer.cls_token_id]+chunks[0]["ids"]+[tokenizer.sep_token_id]:
            raise ValueError("Manual CLS/SEP differs from native tokenizer")
        if [i for chunk in chunks for i in chunk["ids"]] != ids:
            raise ValueError("Incomplete chunk token coverage")
        descriptors[sid] = []
        for chunk in chunks:
            key = hashlib.sha256(json.dumps(chunk["ids"], separators=(",", ":")).encode()).hexdigest()
            if inputs.setdefault(key, chunk["ids"]) != chunk["ids"]:
                raise ValueError("Chunk hash collision")
            descriptors[sid].append(dict(key=key, start=chunk["start"], stop=chunk["stop"], content_tokens=len(chunk["ids"])))
    keys, raw = sorted(inputs), {}
    with torch.inference_mode():
        for start in range(0, len(keys), 32):
            batch_keys = keys[start:start+32]
            batches = [[tokenizer.cls_token_id]+inputs[key]+[tokenizer.sep_token_id] for key in batch_keys]
            width = max(map(len, batches))
            if width > 256 or any(len(batch) != len(inputs[key])+2 for key, batch in zip(batch_keys, batches)):
                raise ValueError("Encoder window exceeded")
            features = dict(input_ids=torch.tensor([ids+[tokenizer.pad_token_id]*(width-len(ids)) for ids in batches], dtype=torch.long),
                attention_mask=torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids in batches], dtype=torch.long),
                token_type_ids=torch.zeros((len(batches), width), dtype=torch.long))
            vectors = model(features)["sentence_embedding"].cpu().numpy()
            if vectors.shape != (len(batch_keys), 384) or not np.isfinite(vectors).all():
                raise ValueError("Invalid encoder vectors")
            raw.update(zip(batch_keys, vectors))
    ids = sorted(snapshots)
    vectors = np.array([pool_chunks([raw[b["key"]] for b in descriptors[sid]],
                                    [b["content_tokens"] for b in descriptors[sid]])
                        if descriptors[sid] else np.zeros(384) for sid in ids]).reshape(len(ids),384)
    output.mkdir(parents=True)
    np.savez_compressed(output/"minilm_embeddings.npz", snapshot_ids=np.array(ids, dtype=str), chunked=vectors)
    save(output/"encoder.json", dict(model=MODEL, revision=REVISION, files=PINNED_FILES, dimensions=384,
         encoder_training=False, tokenizer_class=type(tokenizer).__name__,
         versions={p: importlib.metadata.version(p) for p in ("torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub")},
         snapshots_sha256=hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
         snapshots_canonical_sha256=snapshots_canonical_sha256(snapshots),
         embeddings_sha256=hashlib.sha256((output/"minilm_embeddings.npz").read_bytes()).hexdigest(),
         maximum_tokens=256, content_budget=254, special_tokens=2, overlap=0, decode=False,
         pooling="normalize chunk; content-count weighted mean; normalize document", device="cpu", learned_fits=0,
         documents=descriptors))
    return vectors

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-encode", action="store_true")
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model-dir", type=Path)
    models.add_argument("--download-model", type=Path, help="Explicitly download official pinned files into a fresh guarded directory")
    args = parser.parse_args(argv)
    if not args.allow_encode:
        parser.error("Encoding requires --allow-encode")
    output = guarded(args.output)
    if output.exists():
        parser.error("Use a fresh encoding output directory")
    model_dir = args.model_dir
    if args.download_model:
        model_dir = guarded(args.download_model)
        if model_dir.exists():
            parser.error("Download destination must be fresh")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL, revision=REVISION, allow_patterns=list(PINNED_FILES), local_dir=str(model_dir))
    encode(args.snapshots, model_dir, output, allow_encode=True)

if __name__ == "__main__":
    main()

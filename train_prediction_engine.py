"""
Materials Stack Prediction Engine Training v1.0.0
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import os
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec, phrases
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

#Configuration
CORPUS_FILE   = Path("data/processed/master_corpus.txt")
PHRASES_FILE  = Path("data/processed/master_corpus_phrases.txt")
MODEL_DIR     = Path("models")

WORD2VEC_CONFIG = dict(
    vector_size  = 300,
    window       = 10,
    min_count    = 10,
    sg           = 1,
    hs           = 0,
    negative     = 20,
    epochs       = 10,
    workers      = os.cpu_count(),
    sample       = 1e-5,
    seed         = 42,
)

PHRASES_CONFIG = dict(
    min_count   = 20,
    threshold   = 15,
    delimiter   = "_",
)


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch     = 0
        self.prev_loss = None
        self.t_start   = None

    def on_epoch_begin(self, model):
        self.t_start = time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        delta = f"  delta={loss - self.prev_loss:+.0f}" if self.prev_loss is not None else ""
        elapsed = time.time() - self.t_start
        log.info(
            f"Epoch {self.epoch + 1:02d} — loss={loss:.0f}{delta}  "
            f"({elapsed:.0f}s)"
        )
        # Tip: if delta is < 1% of loss for 2+ consecutive epochs, stop early.
        self.prev_loss = loss
        self.epoch += 1

def build_phrases(corpus_path: Path, out_path: Path) -> None:
    log.info("Building bigram phrase model ...")
    sentences     = LineSentence(str(corpus_path))
    bigram_model  = phrases.Phrases(sentences, **PHRASES_CONFIG)
    bigram        = phrases.Phraser(bigram_model)

    log.info("Building trigram phrase model ...")
    trigram_model = phrases.Phrases(bigram[sentences], **PHRASES_CONFIG)
    trigram       = phrases.Phraser(trigram_model)

    log.info(f"Writing phrase-merged corpus to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sent in trigram[bigram[sentences]]:
            f.write(" ".join(sent) + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bigram_model.save(str(MODEL_DIR / "bigram_phrases.pkl"))
    trigram_model.save(str(MODEL_DIR / "trigram_phrases.pkl"))
    log.info("Phrase models saved.")

def train(use_phrases: bool = True) -> Word2Vec:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = CORPUS_FILE
    if use_phrases:
        if not PHRASES_FILE.exists():
            build_phrases(CORPUS_FILE, PHRASES_FILE)
        corpus_path = PHRASES_FILE

    log.info(f"Training Word2Vec on {corpus_path}")
    log.info(f"Config: {WORD2VEC_CONFIG}")
    log.info(f"CPUs available: {os.cpu_count()}")

    # compute_loss=True enables the LossLogger callback
    model = Word2Vec(
        corpus_file  = str(corpus_path),
        compute_loss = True,
        callbacks    = [LossLogger()],
        **WORD2VEC_CONFIG,
    )

    model_path = MODEL_DIR / "materials_w2v.model"
    vec_path   = MODEL_DIR / "materials_w2v.kv"

    model.save(str(model_path))
    model.wv.save(str(vec_path))

    log.info(f"Vocabulary size : {len(model.wv):,} tokens")
    log.info(f"Model saved     : {model_path}")
    log.info(f"Vectors saved   : {vec_path}")

    return model

def sanity_check(model: Word2Vec) -> None:
    wv = model.wv
    test_terms = [
        "LiFePO4", "cathode", "perovskite", "electrolyte",
        "bandgap", "thermoelectric", "solid_state_electrolyte",
    ]

    log.info("\n── Sanity check: most similar terms ──")
    for term in test_terms:
        t = term.lower()
        if t in wv:
            similar = wv.most_similar(t, topn=5)
            pairs   = ", ".join(f"{w} ({s:.3f})" for w, s in similar)
            log.info(f"  {term:30s} -> {pairs}")
        else:
            log.info(f"  {term:30s} -> NOT IN VOCABULARY")

if __name__ == "__main__":
    model = train(use_phrases=True)
    sanity_check(model)
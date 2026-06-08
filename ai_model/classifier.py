from transformers import pipeline
import time
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info('Loading DistilBERT model...')
        _pipeline = pipeline(
            'sentiment-analysis',
            model=MODEL_NAME,
            device=-1,          # CPU
            truncation=True,
            max_length=512
        )
        logger.info('Model loaded successfully')
    return _pipeline

def predict(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError('Input text cannot be empty')
    clf = _get_pipeline()
    start = time.perf_counter()
    result = clf(text)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        'label': result['label'],
        'score': round(result['score'], 4),
        'inference_time_ms': round(elapsed_ms, 2)
    }

def is_model_loaded() -> bool:
    return _pipeline is not None
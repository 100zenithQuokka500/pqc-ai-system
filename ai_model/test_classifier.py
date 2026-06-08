import pytest
from classifier import predict, batch_predict
 
def test_positive_sentiment():
    result = predict('This product is absolutely amazing and I love it!')
    assert result['label'] == 'POSITIVE'
    assert result['score'] > 0.9
    assert result['inference_time_ms'] > 0
    print(f'Positive test: {result}')
 
def test_negative_sentiment():
    result = predict('The service was terrible. I want a full refund.')
    assert result['label'] == 'NEGATIVE'
    assert result['score'] > 0.8
 
def test_batch_predict():
    texts = ['I love this!', 'This is awful.', 'It was okay.']
    results = batch_predict(texts)
    assert len(results) == 3
    assert results[0]['label'] == 'POSITIVE'
    assert results[1]['label'] == 'NEGATIVE'
 
def test_inference_time_is_realistic():
    result = predict('Testing inference time measurement.')
    # Should take between 10ms and 2000ms on any modern CPU
    assert 10 < result['inference_time_ms'] < 2000
 
def test_empty_text_raises():
    with pytest.raises(ValueError):
        predict('')
 
if __name__ == '__main__':
    test_positive_sentiment()
    test_negative_sentiment()
    test_batch_predict()
    test_inference_time_is_realistic()
    print('All Layer 1 tests passed!')

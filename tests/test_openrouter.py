import pytest
from src.context_switching_attn.openrouter import OpenRouterClient

@pytest.fixture(scope="module")
def client():
    return OpenRouterClient("meta-llama/llama-3.1-8b-instruct")

def test_generate_basic_response(client):
    history = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = client.generate(history)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert any(word.lower() in response.lower() for word in ["paris", "city"])
    print("Generate response:", response)

def test_generate_multi_turn(client):
    history = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "What language do they speak there?"}
    ]
    response = client.generate(history)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert any(word.lower() in response.lower() for word in ["french", "language"])
    print("Multi-turn response:", response)

def test_classify_basic(client):
    history = [
        {"role": "user", "content": "Which animal barks?"}
    ]
    choices = ["Cat", "Dog", "Fish", "Elephant"]
    pred, conf, probs = client.classify(history, choices)

    assert isinstance(pred, int)
    assert 0 <= pred < len(choices)
    assert isinstance(conf, float)
    assert isinstance(probs, list)
    assert len(probs) == len(choices)
    assert abs(sum(probs) - 1.0) < 1e-3
    assert choices[pred].lower() == "dog"
    print(f"Predicted: {choices[pred]} with confidence {conf:.2f}")
    print("Probabilities:", probs)

def test_classify_with_similar_choices(client):
    history = [
        {"role": "user", "content": "Which planet is known for its rings?"}
    ]
    choices = ["Mars", "Earth", "Jupiter", "Saturn"]
    pred, conf, probs = client.classify(history, choices)

    assert isinstance(pred, int)
    assert 0 <= pred < len(choices)
    assert choices[pred].lower() == "saturn"
    print(f"Predicted: {choices[pred]} with confidence {conf:.2f}")

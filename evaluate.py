from transformers import BartTokenizer, BartForConditionalGeneration
from evaluate import load


def evaluate_model(input_text):
    """
    Оценивает модель на одном примере.
    """
    # Загрузка модели и токенизатора
    model_name = "./saved_model"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Генерация суммаризации
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=1024, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150,
                             min_length=30, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Оценка
    rouge = load("rouge")
    results = rouge.compute(predictions=[summary], references=[input_text])
    print(f"Summary: {summary}")
    print(f"ROUGE Scores: {results}")


if __name__ == "__main__":
    # Тестовый пример
    test_text = "CNN reported today that..."
    evaluate_model(test_text)

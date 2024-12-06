from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка модели и токенизатора T5-small
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def abstractive_summary(text, max_length=150, min_length=25):
    """
    Генерация краткого содержания с использованием T5-small.
    """
    # Добавляем префикс задачи суммаризации
    input_text = 'summarize: ' + text

    # Кодирование входного текста
    input_ids = tokenizer.encode(
        input_text, return_tensors='pt', truncation=True)

    # Генерация резюме
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Декодирование и возврат резюме
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

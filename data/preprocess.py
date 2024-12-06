from datasets import load_dataset
from transformers import BartTokenizer


def load_and_prepare_data(tokenizer_name="facebook/bart-large"):
    """
    Загружает данные CNN/DailyMail и применяет токенизацию.
    """
    # Загрузка данных
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    # train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']
    train_data = dataset['train'].shuffle(seed=42).select(
        range(1000))  # Берем первые 1000 примеров примерно 5 часов при макс_лен 256
    val_data = dataset['validation'].shuffle(seed=42).select(
        range(100))  # Берем первые  примеров
    test_data = dataset['test'].shuffle(seed=42).select(
        range(100))  # Берем первые 200 примеров

    # Загрузка токенизатора
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name)

    # Токенизация
    def tokenize_function(batch):
        inputs = tokenizer(
            batch["article"], padding="max_length", truncation=True, max_length=256)
        labels = tokenizer(
            batch["highlights"], padding="max_length", truncation=True, max_length=256)

        inputs["labels"] = labels["input_ids"]  # Добавляем метки отдельно
        return inputs

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    return train_data, val_data, test_data

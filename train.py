from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import torch
from tqdm import tqdm  # Для прогресс-бара
from data.preprocess import load_and_prepare_data
from models.model_utils import load_model


def train_model():
    """
    Тренирует модель на CNN/DailyMail.
    """
    # Подготовка данных
    train_data, val_data, _ = load_and_prepare_data()
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4)

    # Загрузка модели
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Оптимизатор и scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500,
                              num_training_steps=len(train_dataloader) * 3)

    # Обучение
    model.train()
    for epoch in range(3):  # Число эпох
        print(f"Epoch {epoch + 1}")

        # Прогресс-бар для эпохи
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # Преобразуем список тензоров в один тензор с помощью torch.stack()
            input_ids = torch.stack(batch["input_ids"]).to(device)
            attention_mask = torch.stack(batch["attention_mask"]).to(device)
            labels = torch.stack(batch["labels"]).to(device)

            # Передаем данные в модель
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Обновляем прогресс-бар с текущей потерей
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Loss after epoch {epoch + 1}: {loss.item()}")

    # Сохранение модели
    model.save_pretrained(
        "/run/media/wexel/7c741b0e-25f0-4df9-b7ae-24f423064507/home/glooma/Code/Python/ML/NLP_summirize_text/models/bart_pretrained")
    print("Model saved!")


if __name__ == "__main__":
    train_model()

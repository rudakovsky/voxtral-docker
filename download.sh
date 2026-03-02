#!/bin/bash
# Скачивает Voxtral модель из переменной HF_TOKEN

set -e

echo "🚀 Voxtral Model Downloader"
echo "============================="

# Читаем HF_TOKEN из окружения
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Ошибка: Установи HF_TOKEN"
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx"
    echo "  ./download.sh"
    exit 1
fi

CACHE_DIR="./models"
echo "📥 HF_TOKEN: ${HF_TOKEN:0:10}..."
echo "📁 Скачиваем в: $CACHE_DIR"

# Создаём папку
mkdir -p "$CACHE_DIR"

echo "📥 Токенизатор..."
python3 -c "
import os
from transformers import AutoTokenizer
os.environ['HF_TOKEN'] = '$HF_TOKEN'
tokenizer = AutoTokenizer.from_pretrained('mistralai/Voxtral-Mini-4B-Realtime-2602', cache_dir='$CACHE_DIR')
print('✅ Токенизатор готов')
"

echo "📥 Модель (~4GB)..."
python3 -c "
import os
from transformers import AutoModelForSpeechSeq2Seq
import torch
os.environ['HF_TOKEN'] = '$HF_TOKEN'
model_id = 'mistralai/Voxtral-Mini-4B-Realtime-2602'
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    cache_dir='$CACHE_DIR', 
    token='$HF_TOKEN',
    torch_dtype=torch.float32
)
print('✅ МОДЕЛЬ СКАЧАНА!')
"

# Размер папки
SIZE=$(du -sh "$CACHE_DIR" | cut -f1)
echo ""
echo "✅ ВСЁ ГОТОВО!"
echo "📁 Модель: $CACHE_DIR ($SIZE)"
echo "🎉 Теперь: docker build --build-arg HF_TOKEN=\$HF_TOKEN -t voxtral-mini-local ."


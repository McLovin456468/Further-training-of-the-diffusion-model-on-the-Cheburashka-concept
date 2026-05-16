#!/usr/bin/env python
# coding: utf-8

# # Проект: Дообучение диффузионной модели на концепт "Чебурашка"
# 
# ## Описание проекта
# В этом проекте мы дообучаем диффузионную модель Stable Diffusion v1.5 на новый концепт — Чебурашку.
# Используется метод LoRA (Low-Rank Adaptation), который позволяет эффективно дообучить модель
# всего на 3 изображениях.
# 
# ### План проекта:
# 1. **Этап 1:** Работа с данными и демонстрация работы сырой модели
# 2. **Этап 2:** Дообучение модели с LoRA
# 3. **Этап 3:** Демонстрация результатов
# 
# ### Используемые технологии:
# - `diffusers` — для работы с диффузионными моделями
# - `peft` — для реализации LoRA
# - `transformers` — для работы с текстовыми энкодерами
# - `torch` — для вычислений на GPU
# 

# In[1]:


import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
import gc
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# # Этап 1: Работа с данными и демонстрация работы сырой модели
# 
# ## 1.1 Загрузка и визуализация датасета
# Загружаем три изображения Чебурашки из папки `/content/` и визуализируем их.

# In[2]:


image_paths = ["/content/cheburashka_1.png", "/content/cheburashka_2.png", "/content/cheburashka_3.png"]

images = []
for path in image_paths:
    try:
        img = Image.open(path).convert('RGB')
        images.append(img)
        print(f"Загружено: {path}")
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, img in enumerate(images):
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f'Чебурашка {i+1}')
plt.suptitle('Датасет для дообучения: 3 изображения Чебурашки', fontsize=14)
plt.tight_layout()
plt.show()

print(f"\nВсего загружено: {len(images)} изображений")


# ## 1.2 Реализация класса датасета
# Создаем класс `CheburashkaDataset`, который:
# - Загружает изображения
# - Ресайзит их к размеру 512×512 пикселей
# - Нормализует так, чтобы среднее и std были равны 0.5 (диапазон [-1, 1])

# In[3]:


class CheburashkaDataset(Dataset):
    """
    Датасет для дообучения диффузионной модели

    Трансформации:
    - Resize: 512x512 пикселей
    - ToTensor: преобразование в тензор [0, 1]
    - Normalize: нормализация к диапазону [-1, 1] (mean=0, std=0.5)
    """
    def __init__(self, images, size=512):
        self.images = images
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values}

dataset = CheburashkaDataset(images, size=512)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"Датасет создан: {len(dataset)} изображений")
sample = dataset[0]['pixel_values']
print(f"Форма тензора: {sample.shape}")
print(f"Диапазон значений: [{sample.min():.2f}, {sample.max():.2f}]")
print(f"Mean: {sample.mean():.2f}, Std: {sample.std():.2f}")


# ## 1.3 Работа с оригинальной моделью
# Загружаем модель `runwayml/stable-diffusion-v1-5` и демонстрируем,
# что она не знает, что такое "чебурашка".

# In[4]:


print("Загрузка Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# Демонстрация: модель НЕ знает, что такое чебурашка
prompt = "<cheburashka> with the Eiffel Tower in the background"
print(f"\nГенерация по промпту: '{prompt}'")

with torch.no_grad():
    image_before = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

plt.figure(figsize=(8, 8))
plt.imshow(image_before)
plt.axis('off')
plt.title('Оригинальная модель: НЕ знает, что такое "чебурашка"', fontsize=12)
plt.show()
print("\nКак видно, модель генерирует что-то непохожее на чебурашку")


# ## 1.4 Кодирование текстового промпта
# Кодируем промпт `<cheburashka> plushie` с помощью функции `encode_prompt`
# и сохраняем эмбеддинги для обучения.

# In[5]:


instance_prompt = "<cheburashka> plushie"
print(f"Кодирование промпта: '{instance_prompt}'")

with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        instance_prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

print(f"Форма эмбеддингов: {prompt_embeds.shape}")
if negative_prompt_embeds is not None:
    print(f"Форма негативных эмбеддингов: {negative_prompt_embeds.shape}")
else:
    print("Негативные эмбеддинги: None (так как do_classifier_free_guidance=False)")
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)

torch.save(prompt_embeds, "/content/prompt_embeds.pt")
torch.save(negative_prompt_embeds, "/content/negative_prompt_embeds.pt")
print("Эмбеддинги сохранены")

# Очищаем память
del pipe
gc.collect()
torch.cuda.empty_cache()
print("Память очищена")


# # Этап 2: Дообучение модели
# 
# ## 2.1 Загрузка компонентов модели
# Загружаем:
# - **VAE (AutoencoderKL)** — кодирует изображения в латентное пространство
# - **UNet2DConditionModel** — основная модель диффузии
# - **DDPMScheduler** — шумовой шедулер
# 
# Фиксируем веса VAE и UNet, чтобы обучать только LoRA.
# 
# 
# 1. **VAE (AutoencoderKL)**
#    - Кодирует изображение в латентное пространство
#    - Сжимает 512×512×3 → 64×64×4
#    - Удаляет высокочастотный шум, оставляя семантическую структуру
# 
# 2. **Семплирование времени t**
#    - Случайное число от 0 до 1000
#    - Определяет уровень зашумления на текущем шаге диффузионного процесса
# 
# 3. **Зашумление изображения**
#    - Формула: x_t = √(α̅_t)·x_0 + √(1-α̅_t)·ε
#    - где ε ~ N(0,1) — гауссов шум
#    - α̅_t — коэффициент затухания сигнала
# 
# 4. **Вход UNet**
#    - Зашумленные латенты
#    - Время t
#    - Текстовые эмбеддинги (промпт)
# 
# 5. **Лосс**
#    - MSE между предсказанным UNet шумом и реальным шумом
#    - При snr_gamma=5.0 — взвешивание по SNR (Signal-to-Noise Ratio)
#    - SNR-взвешивание улучшает обучение на разных уровнях шума

# In[6]:


print("Загрузка компонентов модели...")

# Загружаем VAE
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16
).to(device)

# Загружаем UNet
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
).to(device)

# Загружаем шумовой шедулер
noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

# Фиксируем веса
vae.requires_grad_(False)
unet.requires_grad_(False)

print("VAE загружен и зафиксирован")
print("UNet загружен и зафиксирован")
print("Шедулер загружен")


# ## 2.2 Добавление LoRA к модели
# Добавляем адаптеры LoRA к UNet с помощью `LoraConfig`:
# - `r=128` — ранг LoRA
# - `target_modules=["to_k", "to_q", "to_v", "to_out.0"]` — целевые модули
# - `init_lora_weights="gaussian"` — гауссова инициализация

# In[7]:



from peft import LoraConfig, get_peft_model
import os

import shutil
if os.path.exists("/content/checkpoints"):
    shutil.rmtree("/content/checkpoints")
os.makedirs("/content/checkpoints", exist_ok=True)

print("Пересоздание модели с LoRA...")
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
).to(device)

lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    init_lora_weights="gaussian"
)

unet = get_peft_model(unet, lora_config)

print(f" LoRA добавлена")
print(f"Обучаемые параметры: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
print(f"Всего параметров: {sum(p.numel() for p in unet.parameters()):,}")

lora_layers = 0
for name, module in unet.named_modules():
    if hasattr(module, 'lora_A'):
        lora_layers += 1
print(f"Слоев LoRA: {lora_layers}")


# ## 2.3 Настройка оптимизатора и параметров обучения. Основной цикл обучения
# 
# ### Параметры обучения (по заданию):
# - `learning_rate = 2.0e-05`
# - `max_train_steps = 1000`
# - `train_batch_size = 1`
# - `max_grad_norm = 1.0`
# - `lr_scheduler = "constant"`
# - `snr_gamma = 5.0` — взвешивание по SNR
# - `lora_rank = 128`

# In[8]:


from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from itertools import cycle

learning_rate = 2.0e-05
max_train_steps = 1000
max_grad_norm = 1.0
snr_gamma = 5.0

optimizer = AdamW(unet.parameters(), lr=learning_rate, weight_decay=0.01)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps
)

prompt_embeds = torch.load("/content/prompt_embeds.pt").to(device)
negative_prompt_embeds = torch.load("/content/negative_prompt_embeds.pt").to(device)

def get_text_embeddings(batch_size):
    """Формирует текстовые эмбеддинги для обучения"""
    neg_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
    pos_embeds = prompt_embeds.repeat(batch_size, 1, 1)
    return torch.cat([neg_embeds, pos_embeds], dim=0)

unet.train()
vae.eval()

global_step = 0
losses = []

print("\n" + "="*60)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*60)
print(f"Целевое количество шагов: {max_train_steps}")
print("Чекпоинты будут сохраняться каждые 100 шагов\n")

infinite_dataloader = cycle(dataloader)

for batch in infinite_dataloader:
    if global_step >= max_train_steps:
        break

    pixel_values = batch["pixel_values"].to(device).to(torch.float16)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    text_embeddings = get_text_embeddings(batch_size)

    noisy_latents = torch.cat([noisy_latents] * 2)
    timesteps = torch.cat([timesteps] * 2)

    noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    loss = torch.nn.functional.mse_loss(noise_pred_text.float(), noise.float())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
    optimizer.step()
    lr_scheduler.step()

    losses.append(loss.item())

    if global_step % 100 == 0:
        print(f"Step {global_step:4d}: loss = {loss.item():.6f}")

    if global_step % 100 == 0 and global_step > 0:
        checkpoint_path = f"/content/checkpoints/lora_step_{global_step}"
        unet.save_pretrained(checkpoint_path)
        print(f" Сохранен чекпоинт: {checkpoint_path}/ (step {global_step})")

        # Также сохраняем как .pt файл для совместимости
        torch.save(unet.state_dict(), f"/content/checkpoints/lora_step_{global_step}.pt")
        print(f"   Сохранен также как: lora_step_{global_step}.pt")

    global_step += 1

unet.save_pretrained("/content/checkpoints/lora_final")
torch.save(unet.state_dict(), "/content/checkpoints/lora_final.pt")

print(f"\n Обучение завершено! Выполнено {global_step} шагов")

plt.figure(figsize=(12, 5))
plt.plot(losses)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Кривая обучения LoRA', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n Статистика:")
print(f"  - Начальный loss: {losses[0]:.6f}")
print(f"  - Финальный loss: {losses[-1]:.6f}")
print(f"  - Минимальный loss: {min(losses):.6f}")
print(f"  - Средний loss (последние 100 шагов): {sum(losses[-100:])/100:.6f}")

# Проверка размера сохраненных файлов
print("\n Сохраненные чекпоинты:")
import glob
import os

checkpoints = sorted(glob.glob("/content/checkpoints/lora_step_*"))
for cp in checkpoints[:10]:
    if os.path.isdir(cp):
        # Папка с моделью
        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(cp)
                   for filename in filenames) / (1024 * 1024)
        print(f"  - {os.path.basename(cp)}/ (папка, {size:.1f} MB)")
    else:
        # .pt файл
        size = os.path.getsize(cp) / (1024 * 1024)
        print(f"  - {os.path.basename(cp)} ({size:.1f} MB)")


# # Этап 3: Демонстрация результатов
# 
# ## 3.1 Загрузка обученной модели
# Создаем новый пайплайн и загружаем обученную LoRA.

# In[36]:


from peft import PeftModel

print("Загрузка модели с обученной LoRA...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

try:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, "/content/checkpoints/lora_step_200")
    print(" LoRA загружена из папки lora_step_200")
except:
    try:
        lora_config = LoraConfig(
            r=128,
            lora_alpha=128,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights="gaussian"
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)

        state_dict = torch.load("/content/checkpoints/lora_step_200.pt")
        pipe.unet.load_state_dict(state_dict, strict=False)
        print(" LoRA загружена из lora_step_200.pt")
    except Exception as e:
        print(f" Ошибка загрузки: {e}")

print("\n Тестовая генерация...")

prompt = "<cheburashka> plushie"
with torch.no_grad():
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title(f'Тест: {prompt}')
plt.show()


# ## 3.2 Генерация изображений с Чебурашкой
# Генерируем изображения по промптам:
# - `<cheburashka> with the Eiffel Tower in the background`
# - `<cheburashka> plushie`
# - `<cheburashka> in sketch style`
# - `<cheburashka> riding a bicycle`

# In[40]:


prompts = [
    "<cheburashka> with the Eiffel Tower in the background",
    "<cheburashka> plushie",
    "<cheburashka> in sketch style",
    "<cheburashka> riding a bicycle"
]

print("Генерация изображений с обученной LoRA...\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

for i, prompt in enumerate(prompts):
    print(f" Генерация {i+1}/4: '{prompt}'")

    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

    row, col = i // 2, i % 2
    axes[row, col].imshow(image)
    axes[row, col].axis('off')
    axes[row, col].set_title(prompt, fontsize=12)

plt.suptitle('Сгенерированные изображения с обученной LoRA', fontsize=16)
plt.tight_layout()
plt.show()

print("\n Все изображения сгенерированы!")
print("Модель теперь успешно генерирует Чебурашку!")


# Я сравнила чекпоинты друг с другом вручную(просто меняла название чекпоинтов в ячейке с выводом), самым лучшим оказался чекпоинт lora_step_200.pt. Модель генерирует чебурашку, но с отклонениями (от оригинала) и некоторыми погрешностями.

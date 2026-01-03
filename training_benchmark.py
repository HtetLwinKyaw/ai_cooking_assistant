# tools/training_benchmark.py
import time, torch, os
from torch.utils.data import DataLoader
from ingredients_recipe.data.dataset import RecipeDataset
from ingredients_recipe.models.model import FullModel
import torch.nn as nn

def run_training_benchmark(data_dir="data/raw", batch_size=4, warmup=2, timed=6, img_size=224):
    ds = RecipeDataset(data_dir=data_dir, build_vocab=False, img_size=img_size, max_recipe_len=32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except: pass

    model = FullModel(encoder_dim=512, num_ingredients=len(ds.ingredient2idx), vocab_size=ds.tokenizer.vocab_size, pretrained_encoder=False)
    model = model.to(device)
    model.train()

    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    it = iter(loader)
    # warmup
    for _ in range(warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        images = batch["image"].to(device)
        ing_targets = batch["ingredients"].to(device)
        optim.zero_grad()
        outputs = model(images, tgt_recipe_ids=None)
        loss = loss_fn(outputs["ingredient_logits"], ing_targets)
        loss.backward()
        optim.step()

    # timed runs
    start = time.time()
    for _ in range(timed):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        images = batch["image"].to(device)
        ing_targets = batch["ingredients"].to(device)

        t0 = time.time()
        optim.zero_grad()
        outputs = model(images, tgt_recipe_ids=None)
        loss = loss_fn(outputs["ingredient_logits"], ing_targets)
        loss.backward()
        optim.step()
        t1 = time.time()
    end = time.time()

    avg_per_batch = (end - start) / timed
    img_per_sec = batch_size / avg_per_batch
    print(f"Batch size: {batch_size}, Avg batch time: {avg_per_batch:.4f}s, Images/sec (training step): {img_per_sec:.2f}")
    return img_per_sec

if __name__ == "__main__":
    run_training_benchmark()

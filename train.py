from models.simple_cnn import SimpleCNN

from dataset import ImageDataset

from config import (
    DATASET_PATH,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

dataset = ImageDataset(DATASET_PATH)

model = SimpleCNN(NUM_CLASSES)

for epoch in range(EPOCHS):

    X, y = dataset.get_batch(BATCH_SIZE)

    loss = model.forward(X, y)

    model.backward(LEARNING_RATE)

    print(f"Epoch {epoch+1}")

    print("Loss:", loss)
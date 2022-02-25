# Active LR
Active LR optimizer for AdamW and RAdam

This is the PyTorch implementation of the Active LR optimizer on top of AdamW and RAdam as we proposed it in "Read the Signs: Towards Invariance to Gradient Descent’s Hyperparameter Values" under review in ICML 2022.

Vanilla Adam
![equation](https://latex.codecogs.com/svg.image?LR%20=%2010%5E%7B-3%7D)
![Adam3s](https://user-images.githubusercontent.com/62418145/155633568-75d0a565-985b-4d6c-8aa6-76b0265e4fd4.png)

![equation](https://latex.codecogs.com/svg.image?LR%20=%2010%5E%7B-5%7D)
![Adam5s](https://user-images.githubusercontent.com/62418145/155633678-c056bd53-96a9-4d73-bc01-8074ca383f3c.png)

![equation](https://latex.codecogs.com/svg.image?LR%20=%2010%5E%7B-5%7D)
![ActiveAdam5s](https://user-images.githubusercontent.com/62418145/155633536-d0e4fc9b-33a1-4019-a1eb-d8b1e0008483.png)

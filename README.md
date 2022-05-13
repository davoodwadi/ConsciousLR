# Read the Signs: Towards Invariance to Gradient Descent’s Hyperparameter Values
Active LR optimizer for AdamW, RAdam, SGD with momentum, and Adabelief (ActiveAdamW, ActiveRAdam, ActiveSGD, and ActiveBelief, respectively).

This is the PyTorch implementation of the Active LR optimizer on top of AdamW, RAdam, SGD with momentum, and Adabelief as we proposed and benchmarked in "Read the Signs: Towards Invariance to Gradient Descent’s Hyperparameter Values" under review in ICML 2022.

## Installing the required python packages and using the ActiveLR version of the optimizers
Although following Algorithm 1 in the paper you can modify existing gradient descent optimizers to their ActiveLR version, we have provided the ActiveLR version of AdamW, RAdam, SGD with momentum, and Adabelief in the *optimizers* subdirectory.

Please use conda to create an environment based on the environment.yml file and copy the *optimizer* directory into the same directory as your python script. 



## Gradient norm distribution for parameterized layers in ResNet-18
Vanilla Adam
LR = 1e-3

![Adam3s](https://user-images.githubusercontent.com/62418145/155633568-75d0a565-985b-4d6c-8aa6-76b0265e4fd4.png)

Vanilla Adam
LR = 1e-5

![Adam5s](https://user-images.githubusercontent.com/62418145/155633678-c056bd53-96a9-4d73-bc01-8074ca383f3c.png)

Active Adam
LR = 1e-5

![ActiveAdam5s](https://user-images.githubusercontent.com/62418145/155633536-d0e4fc9b-33a1-4019-a1eb-d8b1e0008483.png)

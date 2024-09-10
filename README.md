# SFA: Spatial-Frequency Adversarial Attack Method

Adversarial attacks can successfully fool deep neural networks (DNNs) by perturbing the input and adversarial examples help evaluate the defensive capabilities of a DNNs. In black-box scenarios, adversarial examples exhibit low transferability against normally trained and defensive models. In this paper, we propose a Spatial Frequency Adversarial Attack (SFA) in both spatial and frequency domains. Specifically, in the spatial domain, inspired by Stochastic Average Gradient (SAG) optimisation, we leverage historical information to create an initial neighborhood sampling example and then sample nearby it to propose an Average Historical Gradient Sample Method (AHGSM), optimising and stabilizing the gradient update direction while introducing high-frequency perturbations. In the frequency domain, we make a groundbreaking discovery that JPEG compression has different effects on attacking normal training and adversarial training models. Then, by exploring the characteristics of effective adversarial examples in frequency domain distributions, we validate the important hypothesis proposed. Finally, we propose a two-stage attack SFA by integrating JPEG compression as a frequency-based attack with spatial-based AHGSM. Abundant experiments on the ImageNet dataset show that SFA significantly improves the transferability of adversarial examples against both normally and adversarial trained models and has become a state-of-the-art attack in spatial and frequency domains.

![attack](./images/attack.png)

![jpeg](./images/jpeg.png)

## Data

We have provided the experimental data in the dataset folder. If you would like to use other data for the experiment, please organize it according to the format of the dataset folder

### Models

You need to download the weights of different classifier models. For details, please refer to  https://github.com/ylhz/tf_to_pytorch_model. Then put them into 'models' folder.

## Requirements

Use the following command line to configure the environment.

```
pip install -r requirements.txt
```

### Implementation

Using `SFA.py` to implement our SFA, you can run this attack as following

```shell
//first_stage
python SFA.py --output_dir outputs_1  
//second_stage
python SFA.py --input_dir outputs_1 --output_dir outputs_2
```

Using `verify.py` to evaluate the attack success rate

```shell
python verify.py
```

We also provide a baseline implementation method. If you are interested, you can use the attack methods in the baseline folder for experimentation. The usage method is the same as SFA


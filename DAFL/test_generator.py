import torch
import matplotlib.pyplot as plt
from DAFL_train import Generator

generator_path = "cache/models/generator_only.pt"
classifier_path = "cache/models/teacher"
latent_dim = 100


def partial_load(model_cls, model_path, device):
    model = model_cls().to(device)
    model.eval()
    saved_state_dict = torch.load(model_path, map_location=device)
    model_state_dict = model.state_dict()
    # filter state dict
    filtered_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = partial_load(Generator, generator_path, device)
    classifier = torch.load(classifier_path)

    print("generator params:")
    for param in generator.parameters():
        print(param)

    for i in range(10):
        test_rand = torch.randn(1, latent_dim)  # 64 is batch size
        output = generator(test_rand).detach()
        label = classifier(output)
        print("output[0] shape is ", output[0].shape)
        print("label origin: ", label)
        print("label is: ", torch.argmax(label.detach()).numpy())

        plt.imshow(output[0][0].detach().numpy(), cmap='gray')
        plt.show()

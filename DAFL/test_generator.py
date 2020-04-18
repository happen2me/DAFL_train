import torch
import matplotlib.pyplot as plt
from DAFL_train import Generator

generator_path = "cache/models/generator.pt"
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

    print("generator params:")
    for param in generator.parameters():
        print(param)

    for i in range(5):
        test_rand = torch.randn(64, latent_dim)  # 64 is batch size
        output = generator(test_rand)
        print("output[0] shape is ", output[0].shape)

        plt.imshow(output[0][0].detach().numpy())
        plt.show()

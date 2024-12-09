from utils import *
import pickle
from vae import VAE
from stn import SpatialTransformer


def test(model, data):
    model.eval()
    test_loss = []
    for img, label in data:
        img = img.to(device)
        img = stn(img).detach()
        with torch.no_grad():
            output = model(img)
        loss = model.loss_function(*output)
        test_loss.append(loss)

    loss_total = [loss["loss"].cpu().detach() for loss in test_loss]
    recons_loss_total = [
        loss["reconstruction_loss"].cpu().detach() for loss in test_loss
    ]
    kld_loss_total = [loss["kld_loss"].cpu().detach() for loss in test_loss]

    model.train()

    return {
        "loss": sum(loss_total) / len(loss_total),
        "recons_loss": sum(recons_loss_total) / len(recons_loss_total),
        "kld_loss": sum(kld_loss_total) / len(kld_loss_total),
    }


def train(train_set, epochs: int = EPOCHS):
    net = VAE(LATENT_DIMS).to(device)
    net.train()
    optim = torch.optim.AdamW(net.parameters(), lr=LR)
    test_loss = {"base": [], "augmented": []}
    train_loss = []
    for i in range(epochs):
        for img, _ in tqdm(train_set, desc=f"Epoch {i + 1:02}/{epochs}"):
            img = img.to(device)
            img = stn(img).detach()

            optim.zero_grad()

            reconstructed_image, original, mu, log_var = net(img)
            loss = net.loss_function(
                reconstructed_image, original, mu, log_var
            )
            train_loss.append(loss)
            loss["loss"].backward()
            optim.step()

        test_loss["base"].append(test(net, test_base))
        test_loss["augmented"].append(test(net, test_augment))
    return net, test_loss, train_loss


stn = SpatialTransformer().to(device)
stn.load_state_dict(
    torch.load(
        "/Users/henrywilliams/Documents/uni/amml/lab5/src/trained-stn.pt",
        weights_only=True,
    )
)
stn.eval()

if __name__ == "__main__":
    print("Experiment 1:\n")
    print(
        "Investigates the performance of a VAE with a Spatial Transformer Network trained on a dataset without spatial transformations"
    )
    print("Training...\n")
    model, test_loss, train_loss = train(train_base)
    torch.save(model, "E1STN-model.pt")
    pickle.dump(test_loss, open("E1STN-test-loss.pickle", "wb"))
    pickle.dump(train_loss, open("E1STN-train-loss.pickle", "wb"))
    print("Training complete\n")
    print(
        f"Final test loss on base test dataset: {test_loss['base'][-1]['loss'].item()}"
    )
    print(
        f"Classification Accuracy on base dataset: {get_linear_classification_acc(model, train_base, test_base)}"
    )
    print(
        f"Final test loss on augmented test dataset: {test_loss['augmented'][-1]['loss'].item()}"
    )
    print(
        f"Classification Accuracy on augmented dataset: {get_linear_classification_acc(model, train_augment, test_augment)}"
    )

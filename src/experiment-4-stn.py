from utils import *
from stn import SpatialTransformer
import pickle
from cvae import CVAE


def test(model, data):
    model.eval()
    test_loss = []
    for i, (img, label) in enumerate(data):
        img = img.to(device)
        img = stn(img)
        with torch.no_grad():
            output = model(img)
        loss = model.loss_function(*output, i)
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
    net = CVAE(LATENT_DIMS, NUM_CLASSES).to(device)
    net.train()
    optim = torch.optim.AdamW(net.parameters(), lr=LR)
    test_loss = {"base": [], "augmented": []}
    train_loss = []
    for i in range(epochs):
        for idx, (img, _) in tqdm(
            enumerate(train_set),
            desc=f"Epoch {i + 1:02}/{epochs}",
            total=len(train_set),
        ):
            img = img.to(device)
            img = stn(img).detach()

            optim.zero_grad()

            output = net(img)
            loss = net.loss_function(*output, idx)
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
    print("Experiment 4:\n")
    print(
        "Investigates the performance of a Conditional VAE with a Spatial Transformer Network trained on a dataset with spatial transformations\n"
    )
    print("Training...\n")
    model, test_loss, train_loss = train(train_augment)
    torch.save(model, "E4STN-model.pt")
    pickle.dump(test_loss, open("E4STN-test-loss.pickle", "wb"))
    pickle.dump(train_loss, open("E4STN-train-loss.pickle", "wb"))
    print("Training complete\n")
    print(
        f"Final test loss on base test dataset: {test_loss['base'][-1]['loss'].item()}"
    )
    print(
        f"Classification Accuracy on base dataset: {get_linear_classification_acc(model, train_base, test_base, latent_size=LATENT_DIMS * NUM_CLASSES)}"
    )
    print(
        f"Final test loss on augmented test dataset: {test_loss['augmented'][-1]['loss'].item()}"
    )
    print(
        f"Classification Accuracy on augmented dataset: {get_linear_classification_acc(model, train_augment, test_augment, latent_size=LATENT_DIMS * NUM_CLASSES)}"
    )

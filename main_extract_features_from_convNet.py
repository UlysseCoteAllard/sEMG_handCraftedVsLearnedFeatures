import torch
import torch.nn as nn
import torch.optim as optim

from Models.model_training import pre_train_model
from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders


def pretrain_raw_convNet(path):
    participants_dataloaders_train, participants_dataloaders_validation, _ = load_dataloaders(path, batch_size=512)

    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35).cuda()

    # Define Loss functions
    cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()
    cross_entropy_loss_domains = nn.CrossEntropyLoss(reduction='mean').cuda()

    # Define Optimizer
    learning_rate = 0.0404709
    print(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Define Scheduler
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                     verbose=True, eps=precision)

    best_weights = pre_train_model(model=model, cross_entropy_loss_for_class=cross_entropy_loss_classes,
                        cross_entropy_loss_for_domain=cross_entropy_loss_domains, optimizer_class=optimizer, scheduler=scheduler,
                        dataloaders={"train": participants_dataloaders_train,
                                     "val": participants_dataloaders_validation},
                        precision=precision)

    torch.save(best_weights, f="weights/TL_best_weights.pt")

if __name__ == "__main__":
    pretrain_raw_convNet(path="Dataset/processed_dataset")

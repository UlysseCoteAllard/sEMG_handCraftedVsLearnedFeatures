import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import train_regressor
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders

def get_regressor_and_test_it(path_dataset='../Dataset/processed_dataset',
                              path_weights='../weights/TL_best_weights.pt',
                              path_bn_statistics="../weights/bn_statistics.pt"):
    participants_dataloaders_train, participants_dataloaders_validation = load_dataloaders(path_dataset, batch_size=512,
                                                                                           validation_cycle=3,
                                                                                           get_test_set=False,
                                                                                           drop_last=True, shuffle=True)
    # TODO generate Y label with the regression
    for number_of_block in range(6):
        for i in range(len(participants_dataloaders_train)):
            # Define Model
            model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35).cuda()
            best_weights = torch.load(path_weights)
            model.load_state_dict(best_weights)
            list_dictionaries_bn_weights = torch.load(path_bn_statistics)
            BN_weights = list_dictionaries_bn_weights[i]
            model.load_state_dict(BN_weights, strict=False)

            model.transform_to_regressor(layer_to_regress_from=number_of_block)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0404709)

            precision = 1e-8
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            model = train_regressor(model, criterion, optimizer, scheduler, dataloaders={
                "train": participants_dataloaders_train[i], "val": participants_dataloaders_validation[i]},
                                    precision=precision)


if __name__ == "__main__":
    get_regressor_and_test_it(path_dataset='../Dataset/processed_dataset', path_weights='../weights/TL_best_weights.pt',
                              path_bn_statistics="../weights/bn_statistics.pt")

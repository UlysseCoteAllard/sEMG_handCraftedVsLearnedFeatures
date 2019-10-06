import torch

from Models.rawConvNet import Model
from PrepareAndLoadData.load_prepared_dataset_in_dataloaders import load_dataloaders

def extract_learned_features_average(path_datset='../Dataset/processed_dataset',
                                     path_weights='../weights/TL_best_weights.pt',
                                     path_bn_statistics="../weights/bn_statistics.pt"):
    participants_dataloaders_train = load_dataloaders(path_datset, batch_size=512, validation_cycle=None,
                                                      get_test_set=False)
    # Define Model
    model = Model(number_of_class=11, number_of_blocks=6, dropout_rate=0.35).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)
    list_dictionaries_bn_weights = torch.load(path_bn_statistics)

    'Generate the features from the training dataset'
    features_learned = []
    with torch.no_grad():
        for dataset_index, dataset in enumerate(participants_dataloaders_train):
            BN_weights = list_dictionaries_bn_weights[dataset_index]
            model.load_state_dict(BN_weights, strict=False)
            model.eval()
            features_participant = []
            for i, data in enumerate(dataset):
                inputs, _ = data
                inputs = inputs.cuda()
                _, features_batch = model(inputs)
                features_participant.append(features_batch)
            features_learned.append(features_participant)

    'Format the features in the shape: Participant x Examples x Layers'



if __name__ == "__main__":
    extract_learned_features_average(path_datset='../Dataset/processed_dataset',
                                     path_weights='../weights/TL_best_weights.pt',
                                     path_bn_statistics="../weights/bn_statistics.pt")

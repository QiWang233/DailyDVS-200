from os.path import dirname
import argparse
import torch
import tqdm
import os
import pdb
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy,top_k_classes,append_cls_to_list,process_pred_list
from utils.models import Classifier
from utils.dataset import NCaltech101,Ours

def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default="", required=True)
    parser.add_argument("--test_dataset", default="", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    # parser.add_argument("--id", type=int, default=0)
    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.checkpoint)), f"Checkpoint{flags.checkpoint} not found."
    assert os.path.isdir(flags.test_dataset), f"Test dataset directory {flags.test_dataset} not found."

    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.test_dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")

    return flags


if __name__ == '__main__':
    flags = FLAGS()

    # test_dataset = NCaltech101(flags.test_dataset)
    test_dataset = Ours(flags.test_dataset)
    # construct loader, responsible for streaming data to gpu
    test_loader = Loader(test_dataset, flags, flags.device)

    # model, load and put to device
    model = Classifier()
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(flags.device)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0

    pred_list = []
    target_list = []
    sum_top5_accuracy = 0
    print("Test step")
    for events, labels in tqdm.tqdm(test_loader):
        
        with torch.no_grad():
            pred_labels, _ = model(events)
            # exit()
            loss, accuracy,top5_accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels,pred_list,target_list)
            # print(f"pred_labels:{pred_labels} labels:{labels}")

        sum_accuracy += accuracy
        sum_loss += loss

        sum_top5_accuracy += top5_accuracy

    sorted_class_accuracies = top_k_classes(pred_list,target_list)
    test_loss = sum_loss.item() / len(test_loader)
    test_accuracy = sum_accuracy.item() / len(test_loader)
    total_acc = 0
    # for i in sorted_class_accuracies:
    #     total_acc += i[1]
    # txt_path = "/home/wangqi/rpg/id_test.txt"
    # with open(txt_path,'a') as file:
    #     file.write(f"{flags.id} {total_acc / len(sorted_class_accuracies)}\n")
    test_top5_accuracy = sum_top5_accuracy.item() / len(test_loader)

    process_pred_list(pred_list)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy} top5:{test_top5_accuracy}")
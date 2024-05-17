import torch
from collections import Counter

def cross_entropy_loss_and_accuracy(prediction, target,pred_list,target_list):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    top5_accuracy = topk_accuracy(prediction, target,pred_list,target_list, k=5)
    return loss, accuracy,top5_accuracy

def topk_accuracy(prediction, target, pred_list,target_list,k=5):
    _, topk_indices = prediction.topk(k, dim=1)
    # print(f"topk_indices:{topk_indices} target:{target}")
    # exit()

    new_data = {}
    new_data['target'] = int( target[0].item() )
    new_data['pred'] = topk_indices.cpu().numpy().flatten().tolist()[0]
    pred_list.append(new_data)
    # print(f"pred:{topk_indices} target:{target}")
    # print(f"pred:{topk_indices.cpu().numpy().flatten().tolist()}")
    # exit()
    # pred_list.append(int(topk_indices[0][0].item()))
    # target_list.append(int(target[0].item()))

    correct = topk_indices.eq(target.view(-1, 1).expand_as(topk_indices))
    topk_accuracy = correct.sum().float() / target.size(0)
    
    return topk_accuracy

def top_k_classes(pred, target):
    class_counts = Counter(target)
    correct_counts = {cls: sum(1 for p, t in zip(pred, target) if p == t == cls) for cls in class_counts}
    class_accuracies = {cls: correct_counts[cls] / class_counts[cls] if class_counts[cls] > 0 else 0 for cls in class_counts}
    sorted_class_accuracies = [(cls, accuracy) for cls, accuracy in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)]
    sum_acc = 0
    for i in sorted_class_accuracies:
        sum_acc += i[1]

    # print(sum_acc / len(sorted_class_accuracies))

    return sorted_class_accuracies

def append_cls_to_list(data,cls_list):
    for cls_data in cls_list:
        if int(data['target']) == int(cls_data['target']):
            if data['pred'] != data['target']:
                cls_data['wrong_cls'].append(data['pred'])
            return
    
    new_data = {}
    new_data['target'] = int(data['target'])
    new_data['wrong_cls'] = []
    if data['pred'] != data['target']:
        new_data['wrong_cls'].append(data['pred'])  

    cls_list.append(new_data)

def process_pred_list(pred_list):
    cls_list = []
    for pred in pred_list:
        append_cls_to_list(pred,cls_list)

    for data in cls_list:
        d = dict(Counter(data['wrong_cls']))
        data['wrong_cls'] = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
 
    out_list = [138,117,120,112,153,14,89,125,189,26]
    for data in cls_list:
        if data['target'] in out_list:

            print(data)

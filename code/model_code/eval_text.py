# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json

from sklearn.metrics import confusion_matrix

from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, \
    Bn_Controller

import os
import pdb
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
from sklearn.metrics import f1_score
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--algorithm', type=str, default=16000)

    args = parser.parse_args()

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']


    fw=open('select_maybe.json','w')

    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)

    if 'remix' in checkpoint_path or 'mine' in checkpoint_path:
        load_state_dict = {key.replace('backbone.', '', 1): value for key, value in load_state_dict.items() if key.startswith('backbone.')}
    # load_state_dict = {key.replace('backbone.', '', 1): value for key, value in load_state_dict.items() if
    #                    key.startswith('backbone.')}

    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    # specify these arguments manually 
    args.num_labels = 40
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    # eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)

    eval_loader = get_data_loader(args,
                                  dataset_dict['eval'],
                                  args.batch_size,
                                  # make sure data_sampler is None for evaluation
                                  data_sampler=None,
                                  num_workers=4,
                                  drop_last=False,
                                  shuffle=False)

    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    epoch=0
    with torch.no_grad():
        for data in tqdm(eval_loader):
            image = data['x_lb']
            image['input_ids'] = image['input_ids'].to('cuda')
            image['attention_mask'] = image['attention_mask'].to('cuda')

            target = data['y_lb']

            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            acc += pred.cpu().eq(target).numpy().sum()
            for index in range(len(pred)):
                content=dataset_dict['eval'][epoch*128+index]
                content['predict']=int(pred[index].cpu())
                fw.write(json.dumps(content)+'\n')
            epoch+=1
            test_feats.append(feat.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_probs.append(prob.cpu().numpy())
            test_labels.append(target.cpu().numpy())
    pdb.set_trace()

    print(torch.bincount(target) / len(target))
    print(torch.bincount(pred) / len(pred))
    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)

    cm = confusion_matrix(test_labels, test_preds,normalize='true')
    print(cm)
    f1_scor=f1_score(test_labels,test_preds, average='macro')
    print(f"F1: {f1_scor}")
    f1_scor = f1_score(test_labels, test_preds, average='weighted')
    print(f"F1: {f1_scor}")

    print(f"Test Accuracy: {acc / len(eval_dset)}")


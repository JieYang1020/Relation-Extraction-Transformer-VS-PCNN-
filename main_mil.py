# -*- coding: utf-8 -*-

from config import opt
import models
import dataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import save_pr, now, eval_metric
import pandas as pd
import csv
import sys

csv.field_size_limit(500 * 1024 * 1024)
train_content_path = r'C:\Users\lenovo\Desktop\Josie\自学\Pytorch_关系抽取\python方法\pytorch-relation-extraction-master\dataset\FilterNYT\train\new_train.csv'
train_label_path = r'C:\Users\lenovo\Desktop\Josie\自学\Pytorch_关系抽取\python方法\pytorch-relation-extraction-master\dataset\FilterNYT\train\new_label.csv'
train_df = pd.read_csv(train_content_path, engine='python')
train_label_df = pd.read_csv(train_label_path, engine='python')

# test_content_path = r'C:\Users\lenovo\Desktop\Josie\自学\Pytorch_关系抽取\python方法\pytorch-relation-extraction-master\dataset\FilterNYT\test\new_test.csv'
# test_label_path = r'C:\Users\lenovo\Desktop\Josie\自学\Pytorch_关系抽取\python方法\pytorch-relation-extraction-master\dataset\FilterNYT\test\new_label.csv'
# test_df = pd.read_csv(test_content_path, engine='python')
# test_label_df = pd.read_csv(test_label_path, engine='python')

def get_new_data(content_df, label_df, train_label):
    # 训练数据集转换(一个格子为一个bag,个数不一致，保留第一个)
    sentence = []
    position = []
    for i in range(content_df.shape[0]):
        temp_content = train_df.iloc[i, 0].replace('[', '').replace(']', '').replace(' ', '').split(',')
        temp_content = [int(j) for j in temp_content]
        sentence.append(temp_content)
    for i in range(content_df.shape[0]):
        temp_content = train_df.iloc[i, 1].replace('[', '').replace(']', '').replace(' ', '').split(',')
        temp_content = [int(j) for j in temp_content]
        position.append(temp_content)
    sentence = torch.tensor(np.array(sentence))
    position = torch.tensor(np.array(position))
    sentence = sentence.unsqueeze(1)
    position = position.view(position.size(0), -1, 82)
    content = torch.cat((sentence, position), 1)
    label = torch.tensor(np.array(label_df['label']))
    content = torch.tensor(np.array(content))
    deal_dataset = TensorDataset(content, label)
    if train_label:
        print('train data大小为: {}'.format(label_df.shape[0]))
    else:
        print('test data大小为: {}'.format(label_df.shape[0]))
    return deal_dataset

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

# def test(**kwargs):
#     pass

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(**kwargs):
    setup_seed(opt.seed)

    kwargs.update({'model': 'Transformer_self'})
    opt.parse(kwargs)

    torch.manual_seed(opt.seed)
    model = getattr(models, 'Transformer_self')(opt)
    deal_dataset = get_new_data(train_df, train_label_df, train_label=True)
    train_data_loader = DataLoader(dataset=deal_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)


    DataModel = getattr(dataset, opt.data + 'Data')
    #
    # # loading data
    # train_data = DataModel(opt.data_root, train=True)
    # train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    # test_data = DataModel(opt.data_root, train=False)
    # test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    # print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-5)
    # optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), rho=0.9, weight_decay=opt.weight_decay)
    for epoch in range(opt.num_epochs):
        for idx, (inputs, labels) in enumerate(train_data_loader):
            labels = labels.float()
            out = model(inputs)
            optimizer.zero_grad()
            labels = labels.long()
            labels = labels.detach()
            # 真实数据的label的size为[batch_size]
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            print("第%d个epoch的第%d个index的loss值%s" % (epoch, idx, loss))
            print("-----")

            '''
            label = [l[0] for l in label_set]
            label = torch.LongTensor(label)
            data = select_instance(model, data, label)
            model.batch_size = opt.batch_size
            optimizer.zero_grad()
            out = model(data)
            # out = model(data, train=True)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            print("第%d个epoch的第%d个index的loss值%s" % (epoch, idx, loss))
            total_loss += loss.item()
        if epoch < -1:
            continue
        '''
        # true_y, pred_y, pred_p = predict(model, test_data_loader)
        # all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)
        #
        # last_pre, last_rec = all_pre[-1], all_rec[-1]
        # print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, last_pre, last_rec))

#(model, data, label)
def select_instance(model, batch_data, labels):

    model.eval()
    select_ent = []
    select_num = []
    select_sen = []
    select_pf = []
    select_pool = []
    select_mask = []
    for idx, bag in enumerate(batch_data):
        print(type(batch_data))
        print("------")
        insNum = bag[1]
        label = labels[idx]
        max_ins_id = 0
        if insNum > 1:
            model.batch_size = insNum
            data = map(lambda x: torch.LongTensor(x), bag)
            out = model(data)
            max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]
            if opt.use_gpu:
                #  max_ins_id = max_ins_id.data.cpu().numpy()[0]
                max_ins_id = max_ins_id.item()
            else:
                #不确定的修改：max_ins_id输出必须为整型
                max_ins_id = max_ins_id.data.numpy()
        max_sen = bag[2][max_ins_id]
        max_pf = bag[3][max_ins_id]
        max_pool = bag[4][max_ins_id]
        max_mask = bag[5][max_ins_id]

        select_ent.append(bag[0])
        select_num.append(bag[1])
        select_sen.append(max_sen)
        select_pf.append(max_pf)
        select_pool.append(max_pool)
        select_mask.append(max_mask)
    # if opt.use_gpu:
    #     data = map(lambda x: torch.LongTensor(x).cuda(), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    # else:
    data = map(lambda x: torch.LongTensor(x), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    model.train()
    return data


def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, (data, labels) in enumerate(test_data_loader):
        true_y.extend(labels)
        for bag in data:
            insNum = bag[1]
            model.batch_size = insNum
            # if opt.use_gpu:
            #     data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            # else:
            data = map(lambda x: torch.LongTensor(x), bag)

            out = model(data)
            out = F.softmax(out, 1)
            max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
            tmp_prob = -1.0
            tmp_NA_prob = -1.0
            pred_label = 0
            pos_flag = False

            for i in range(insNum):
                if pos_flag and max_ins_label[i] < 1:
                    continue
                else:
                    if max_ins_label[i] > 0:
                        pos_flag = True
                        if max_ins_prob[i] > tmp_prob:
                            pred_label = max_ins_label[i]
                            tmp_prob = max_ins_prob[i]
                    else:
                        if max_ins_prob[i] > tmp_NA_prob:
                            tmp_NA_prob = max_ins_prob[i]

            if pos_flag:
                pred_p.append(tmp_prob)
            else:
                pred_p.append(tmp_NA_prob)

            pred_y.append(pred_label)

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(true_y) == size

    model.train()
    return true_y, pred_y, pred_p

if __name__ == "__main__":
    train()
    # import fire
    # fire.Fire()

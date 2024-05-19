import torch
from torch import nn
import sys
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
# Construct the model
####################################################################
def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    # 优化器改进
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader):
            # 小规模输入截断
            if i_batch > 10:
                break
            # print("Batch:", i_batch)
            sample_ind, seqs, quas = batch_X
            # 如果seqs是Tensor，确保使用PyTorch的方法
            if isinstance(seqs, torch.Tensor):
                seqs_is_zero = torch.all(seqs == 0).item()  # .item() 将单元素张量转换为Python的布尔值
            else:
                # 如果seqs是numpy数组，现有的方法是正确的
                seqs_is_zero = np.all(seqs == 0)

            if isinstance(quas, torch.Tensor):
                quas_is_zero = torch.all(quas == 0).item()
            else:
                quas_is_zero = np.all(quas == 0)

            if seqs_is_zero or quas_is_zero:
                continue

            # 由于实际的batch_size=1，所以需要对第一个维度折叠
            seqs = seqs.squeeze(0)
            quas = quas.squeeze(0)
            eval_attr = batch_Y.squeeze(0)

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    seqs, quas, eval_attr = seqs.cuda(), quas.cuda(), eval_attr.cuda()
                    # iemocap不存在
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = seqs.size(0)
            # print("Batch size:", batch_size)
            batch_chunk = hyp_params.batch_chunk

            # CTC

            combined_loss = 0
            # net = nn.DataParallel(model) if batch_size > 10 else model
            net = model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                seqs_chunks = seqs.chunk(batch_chunk, dim=0)
                quas_chunks = quas.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                for i in range(batch_chunk):
                    seqs_i, quas_i = seqs_chunks[i], quas_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(seqs_i, quas_i)

                    # iemocap不存在
                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i  # + ctc_loss
                    raw_loss_i.backward()
                combined_loss = raw_loss
            else:
                preds, hiddens = net(seqs, quas)
                # iemocap不存在
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                # print("Predictions shape:", preds.shape)
                # print("Eval attribute shape:", eval_attr.shape)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss     # + ctc_loss
                combined_loss.backward()

            # if ctc_criterion is not None:
            #     torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
            #     torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
            #     ctc_a2l_optimizer.step()
            #     ctc_v2l_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(loader):
                if i_batch > 1:
                    break
                sample_ind, seqs, quas = batch_X
                # 由于实际的batch_size=1，所以需要对第一个维度折叠
                seqs = seqs.squeeze(0)
                quas = quas.squeeze(0)
                eval_attr = batch_Y.squeeze(0)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        seqs, quas, eval_attr = seqs.cuda(), quas.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = seqs.size(0)

                # if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                #     ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                #     ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                #     audio, _ = ctc_a2l_net(audio)  # audio aligned to text
                #     vision, _ = ctc_v2l_net(vision)  # vision aligned to text

                # net = nn.DataParallel(model) if batch_size > 10 else model
                net = model
                preds, _ = net(seqs, quas)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                preds = preds.unsqueeze(0)
                eval_attr = eval_attr.unsqueeze(0)
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / 2   # (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss,
                                                                                             test_loss))
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    _, results, truths = evaluate(model, criterion, test=True)
    print("results shape:", results.shape)
    print("results:", results)
    # results_softmax = torch.nn.functional.softmax(results.float(), dim=-1)
    # print("after softmax shape:", results_softmax.shape)
    # print("after softmax:", results_softmax[0])

    print("truths shape:", truths.shape)
    print("truths:", truths)
    # truths_softmax = torch.nn.functional.softmax(truths.float(), dim=-1)
    # print("truth after softmax:", truths_softmax[0])
    # if hyp_params.dataset == "mosei_senti":
    #     eval_mosei_senti(results, truths, True)
    # elif hyp_params.dataset == 'mosi':
    #     eval_mosi(results, truths, True)
    # elif hyp_params.dataset == 'iemocap':
    #     eval_iemocap(results, truths)
    # TODO：待补充的数据集
    # hyp_params.dataset == '':
    print("自己设置的模型")
    eval_dna(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

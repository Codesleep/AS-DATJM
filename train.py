# -*- coding: utf-8 -*-
import math
import argparse
import torch
import os
import pickle
import torch.nn as nn
import random
import numpy as np
from sklearn import metrics
from data_utils import ABSADatesetReader, Tokenizer
from bucket_iterator import BucketIterator
from models import ATLSGCN, ATLSGCNDT

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt)
        self._print_args()
        self.global_f1 = 0.

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def run(self, repeats=5):
        #loss
        aspect_criterion = nn.BCELoss()
        emotion_criterion = nn.NLLLoss()

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt', 'w', encoding='utf-8')

        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            max_test_f1, max_R, max_P = self._train(aspect_criterion, emotion_criterion, optimizer)
            print('max_test_f1: {0}     max_R: {1}     max_P: {2}'.format(max_test_f1, max_R, max_P))
            f_out.write('max_test_f1: {0}     max_R: {1}     max_P: {2}'.format(max_test_f1, max_R, max_P))
            print('#' * 100)

        f_out.close()

    def _train(self, aspect_criterion, emotion_criterion, optimizer):
        max_R = 0
        max_P = 0
        max_test_f1 = 0
        global_step = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            aspect_correct, aspect_total = 0, 0
            emotion_correct, emotion_total = 0, 0

            sum = 0
            aspect_train_acc_all = 0
            aspect_test_acc_all = 0
            emotion_train_acc_all = 0
            emotion_test_acc_all = 0
            f1_avg = 0
            R_avg = 0
            P_avg = 0
            aspect_f1_avg = 0
            aspect_R_avg = 0
            aspect_P_avg = 0
            emotion_f1_avg = 0
            emotion_R_avg = 0
            emotion_P_avg = 0

            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()

                optimizer.zero_grad()

                inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                aspect_targets = sample_batched['aspect_label']
                emotion_targets = sample_batched['polarity']
                aspect, emotion, alpha = self.model(inputs)
                loss1 = aspect_criterion(aspect.squeeze(1), aspect_targets.float())#torch.reshape(aspect_targets, (1, 1))
                loss2 = emotion_criterion(emotion, emotion_targets.long())
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    sum = sum + 1
                    aspect_correct += torch.eq(torch.round(aspect).squeeze(1), aspect_targets).sum().item()
                    aspect_total += len(aspect)
                    aspect_train_acc = aspect_correct / aspect_total
                    aspect_train_acc_all = aspect_train_acc_all + aspect_train_acc

                    emotion_correct += (torch.argmax(emotion, -1) == emotion_targets).sum().item()
                    emotion_total += len(emotion)
                    emotion_train_acc = emotion_correct / emotion_total
                    emotion_train_acc_all = emotion_train_acc_all + emotion_train_acc

                    aspect_test_acc, emotion_test_acc, test_F1, R, P , aspect_P, aspect_R, aspect_f1, emotion_P, emotion_R, emotion_f1 = self._evaluate_acc_f1()
                    aspect_test_acc_all = aspect_test_acc_all + aspect_test_acc
                    emotion_test_acc_all = emotion_test_acc_all + emotion_test_acc
                    f1_avg = f1_avg + test_F1
                    R_avg = R_avg + R
                    P_avg = P_avg + P
                    aspect_f1_avg = aspect_f1_avg + aspect_f1
                    aspect_R_avg = aspect_R_avg + aspect_R
                    aspect_P_avg = aspect_P_avg + aspect_P
                    emotion_f1_avg = emotion_f1_avg + emotion_f1
                    emotion_R_avg = emotion_R_avg + emotion_R
                    emotion_P_avg = emotion_P_avg + emotion_P

                    print('loss: {:.4f}, aspect_train_acc: {:.4f}, emotion_train_acc: {:.4f}, aspect_test_acc: {:.4f}, '
                          'emotion_test_acc: {:.4f}, test_F1: {:.4f}, R: {:.4f}, P: {:.4f}, aspect_f1: {:.4f}, aspect_R:'
                          ' {:.4f}, aspect_P: {:.4f}, emotion_f1: {:.4f}, emotion_R: {:.4f}, emotion_P: {:.4f}'
                          .format(loss.item(), aspect_train_acc, emotion_train_acc, aspect_test_acc, emotion_test_acc,
                                  test_F1, R, P, aspect_f1, aspect_R, aspect_P, emotion_f1, emotion_R, emotion_P))
                    if test_F1 > max_test_f1:
                        max_test_f1 = test_F1
                        max_P = P
                        max_R = R
                        if self.opt.save and test_F1 > self.global_f1:
                            self.global_f1 = test_F1
                            torch.save(self.model.state_dict(),
                                       'state_dict/' + self.opt.model_name + '_emo_DG_2_15lr' + self.opt.dataset + '.pkl')
                            print('>>> best model saved.')


            print('aspect_train_acc_avg: {:.4f}, aspect_test_acc_avg: {:.4f}'.format(aspect_train_acc_all / sum,
                                                                                 aspect_test_acc_all / sum))
            print('emotion_train_acc_avg: {:.4f}, emotion_test_acc_avg: {:.4f}, F1_avg: {:.4f}, R_avg: {:.4f}, P_avg: {:.4f}, aspect_f1_avg: {:.4f}, aspect_R_avg: {:.4f}, aspect_P_avg: {:.4f}, emotion_f1_avg: {:.4f}, emotion_R_avg: {:.4f}, emotion_P_avg: {:.4f}'.format(emotion_train_acc_all / sum,
                                                                                       emotion_test_acc_all / sum, f1_avg / sum, R_avg / sum, P_avg / sum, aspect_f1_avg / sum, aspect_R_avg / sum, aspect_P_avg / sum, emotion_f1_avg / sum, emotion_R_avg / sum, emotion_P_avg / sum))
        return max_test_f1, max_R, max_P

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()

        aspect_test_correct, aspect_test_total = 0, 0
        emotion_test_correct, emotion_test_total = 0, 0
        aspect_targets_all, aspect_outputs_all = None, None
        emotion_targets_all, emotion_outputs_all = None, None


        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
                t_aspect_targets = t_sample_batched['aspect_label']
                t_emotion_targets = t_sample_batched['polarity']

                t_aspect, t_emotion, t_alpha = self.model(t_inputs)
                aspect_test_correct += torch.eq(torch.round(t_aspect).squeeze(1), t_aspect_targets).sum().item()
                aspect_test_total += len(t_aspect)
                emotion_test_correct += (torch.argmax(t_emotion, -1) == t_emotion_targets).sum().item()
                emotion_test_total += len(t_emotion)

                if emotion_targets_all is None:
                    aspect_targets_all = t_aspect_targets
                    aspect_outputs_all = t_aspect
                    emotion_targets_all = t_emotion_targets
                    emotion_outputs_all = t_emotion
                else:
                    aspect_targets_all = torch.cat((aspect_targets_all, t_aspect_targets), dim=0)
                    aspect_outputs_all = torch.cat((aspect_outputs_all, t_aspect), dim=0)
                    emotion_targets_all = torch.cat((emotion_targets_all, t_emotion_targets), dim=0)
                    emotion_outputs_all = torch.cat((emotion_outputs_all, t_emotion), dim=0)

            aspect_test_acc = aspect_test_correct / aspect_test_total
            emotion_test_acc = emotion_test_correct / emotion_test_total
            P, R, f1, aspect_P, aspect_R, aspect_f1, emotion_P, emotion_R, emotion_f1 = self.Micro(aspect_outputs_all,
                                                                                                   emotion_outputs_all,
                                                                                                   aspect_targets_all,
                                                                                                   emotion_targets_all)

            return aspect_test_acc, emotion_test_acc, f1, R, P, aspect_P, aspect_R, aspect_f1, emotion_P, emotion_R, emotion_f1

    def sequence_to_text(self, sequence):
        with open(opt.dataset + '_word2idx.pkl', 'rb') as f:
            idx2word = pickle.load(f)
            tokenizer = Tokenizer(word2idx=idx2word)
            texts = []
        for i in range(len(sequence)):
            texts.append(tokenizer.idx2word[sequence[i]])

        return texts

    def Micro(self, aspect_prediction, emotion_prediction, aspect_actual, emotion_actual):
        # aspect_num = 8
        aspect_emotion_num = [0, 1, 2]
        aspect_prediction = torch.round(aspect_prediction).squeeze(1).numpy()
        aspect_actual = aspect_actual.numpy()
        emotion_prediction = torch.argmax(emotion_prediction, -1).numpy()
        emotion_actual = emotion_actual.numpy()

        prediction = torch.tensor(self.Transform(aspect_prediction, emotion_prediction))
        actual = torch.tensor(self.Transform(aspect_actual, emotion_actual))
        ruler = torch.ones_like(prediction)
        # 总共的Micro_F1
        TP_sum = 0
        FP_sum = 0
        FN_sum = 0

        for i in range(len(aspect_emotion_num)):
            x = aspect_emotion_num[i] * ruler
            TP, FP, FN = 0, 0, 0

            TP = ((prediction == x) & (actual == x)).sum()
            FP = ((prediction == x) & (actual != x)).sum()
            FN = ((prediction != x) & (actual == x)).sum()

            TP_sum = TP_sum + TP
            FP_sum = FP_sum + FP
            FN_sum = FN_sum + FN

        P = TP_sum / (TP_sum + FP_sum)
        R = TP_sum / (TP_sum + FN_sum)

        f1 = (2 * P * R) / (P + R)

        # aspect_Micro_F1
        aspect_TP_sum = 0
        aspect_FP_sum = 0
        aspect_FN_sum = 0
        aspect_num = [1]

        for i in range(len(aspect_num)):
            aspect_x = aspect_num[i] * ruler
            TP, FP, FN = 0, 0, 0

            TP = ((torch.tensor(aspect_prediction) == aspect_x) & (torch.tensor(aspect_actual) == aspect_x)).sum()
            FP = ((torch.tensor(aspect_prediction) == aspect_x) & (torch.tensor(aspect_actual) != aspect_x)).sum()
            FN = ((torch.tensor(aspect_prediction) != aspect_x) & (torch.tensor(aspect_actual) == aspect_x)).sum()
            aspect_TP_sum = aspect_TP_sum + TP
            aspect_FP_sum = aspect_FP_sum + FP
            aspect_FN_sum = aspect_FN_sum + FN

        aspect_P = aspect_TP_sum / (aspect_TP_sum + aspect_FP_sum)
        aspect_R = aspect_TP_sum / (aspect_TP_sum + aspect_FN_sum)
        aspect_f1 = (2 * aspect_P * aspect_R) / (aspect_P + aspect_R)

        # emotion_Micro_F1
        emotion_TP_sum = 0
        emotion_FP_sum = 0
        emotion_FN_sum = 0
        emotion_num = [0, 1, 2]

        for i in range(len(emotion_num)):
            emotion_x = emotion_num[i] * ruler
            TP, FP, FN = 0, 0, 0

            TP = ((torch.tensor(emotion_prediction) == emotion_x) & (torch.tensor(emotion_actual) == emotion_x)).sum()
            FP = ((torch.tensor(emotion_prediction) == emotion_x) & (torch.tensor(emotion_actual) != emotion_x)).sum()
            FN = ((torch.tensor(emotion_prediction) != emotion_x) & (torch.tensor(emotion_actual) == emotion_x)).sum()
            emotion_TP_sum = emotion_TP_sum + TP
            emotion_FP_sum = emotion_FP_sum + FP
            emotion_FN_sum = emotion_FN_sum + FN

        emotion_P = emotion_TP_sum / (emotion_TP_sum + emotion_FP_sum)
        emotion_R = emotion_TP_sum / (emotion_TP_sum + emotion_FN_sum)
        emotion_f1 = (2 * emotion_P * emotion_R) / (emotion_P + emotion_R)


        return P, R, f1, aspect_P, aspect_R, aspect_f1, emotion_P, emotion_R, emotion_f1

    def Transform(self, aspect, emotion):
        z = []
        x_label = [1, 1, 1, 1, 0, 0, 0, 0]
        y_label = [0, 1, 2, 3, 0, 1, 2, 3]

        for i in range(len(aspect)):
            for j in range(len(x_label)):
                if ((aspect[i] == x_label[j]) & (emotion[i] == y_label[j])):
                    z.append(j)
                    break

        return z

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='atlsgcn', type=str)
    parser.add_argument('--dataset', default='rest15', type=str, help='rest15, lap15, rest16, lap16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=72, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--aspect_polarity', default=1, type=int)
    parser.add_argument('--emotion_polarity', default=4, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    #parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'atlsgcn': ATLSGCN,
        'atlsgcndt': ATLSGCNDT,
    }

    input_colses = {
        'atlsgcn': ['text_indices', 'aspect_indices', 'dependency_graph'],
        'atlsgcndt': ['text_indices', 'aspect_indices', 'dependency_tree'],
    }

    #~~
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    #~~

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()

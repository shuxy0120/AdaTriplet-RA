import random
import time
import torch
import os
import math
import ipdb
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gumbel import gumbel_softmax

def eval_func(qf, gf, q_pids, g_pids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    qf = normalize(qf, -1)
    gf = normalize(gf, -1)
    q_pids = torch.tensor(q_pids).int().cpu().numpy()
    g_pids = torch.tensor(g_pids).int().cpu().numpy()
    q_camids = np.ones(q_pids.shape[0])
    g_camids = np.zeros(q_pids.shape[0])
    distmat = 1. - qf.mm(gf.t()).cpu().numpy()
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if num_rel != 0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = 0
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    if not all_AP:
        mAP = 0
    # all_cmc = 0
    else:
        mAP = np.asarray(all_AP).astype(np.float32)
        # mmAP = np.mean(mAP)
    return torch.tensor(mAP), indices




def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    # m, n = x.size(0), y.size(0)
    # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = 1 - x.mm(y.t())
    return dist


def hard_example_mining(dist_mat, labels_s, labels_t, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    mask = torch.ones((N, N)).cuda()
    # shape [N, N]
    is_pos = labels_s.expand(N, N).eq(labels_t.expand(N, N).t()).float()
    is_neg = labels_s.expand(N, N).ne(labels_t.expand(N, N).t()).float()

    # positive = mask[is_pos]    ###nagive is zero
    # negative = mask*is_neg     ### positive is zero

    # negative[negative == 0] = 10e10

    dist_ap, _ = torch.max((dist_mat * is_pos).contiguous().view(N, -1), 1, keepdim=True)
    # dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # neg_mid = (dist_mat * is_neg).contiguous().view(N, -1)
    # dist_an  = torch.topk(neg_mid, k=neg_mid.size(1), dim = 1, largest=False)[0].view(N,-1)
    # non_negative = (dist_an>0).float()
    # p, first_non_negative = non_negative.max(1)
    # assert (p.all()==True)
    # dist_an_final = dist_an.gather(1, first_non_negative.view(-1,1))
    # shape [N]
    # dist_an_final = torch.zeros(dist_ap.size()).cuda()
    # dist_an = (dist_mat * is_neg)
    # for n in range(N):
    #    if dist_an[n,:].sum() == 0:
    #        dist_an_final[n,:] = 0
    #    else:
    #        mid = dist_an[n, (dist_an[n, :] != 0)]
    #        dist_an_final[n, :], _ = torch.min(mid,0, keepdim = True)

    # dist_an_final = torch.mean(dist_an.contiguous().view(N, -1), 1, keepdim=True)
    dist_an_final = torch.mean((dist_mat * is_neg).contiguous().view(N, -1), 1, keepdim=True)
    return dist_ap.squeeze(), dist_an_final.squeeze()


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = 1
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat_s, feat_t, labels_s, labels_t, certainty, normalize_feature=False):
        feat_t = normalize(feat_t, -1)
        feat_s = normalize(feat_s, -1)
        dist_mat = euclidean_dist(feat_s, feat_t)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels_s, labels_t)
        loss = (1. - certainty) * F.relu(dist_ap - dist_an + self.margin - certainty/2.)
        # loss = F.relu(dist_ap - dist_an + self.margin - certainty/2.)
        loss = loss.mean()
        return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, gradient_mask):
        input = (input).contiguous()
        output = - (input * reward.unsqueeze(1))
        loss = torch.mean(output)
        return loss


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class TargetDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(TargetDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, self.num_classes:].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, self.num_classes:].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, self.num_classes:].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, self.num_classes:].sum(1).log().mean())
        return loss


class SourceDiscrimLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, num_classes=31):
        super(SourceDiscrimLoss, self).__init__(weight, size_average)
        self.num_classes = num_classes

    def forward(self, input):
        batch_size = input.size(0)
        prob = F.softmax(input, dim=1)

        if (prob.data[:, :self.num_classes].sum(1) == 0).sum() != 0:  ########### in case of log(0)
            soft_weight = torch.FloatTensor(batch_size).fill_(0)
            soft_weight[prob[:, :self.num_classes].sum(1).data.cpu() == 0] = 1e-6
            soft_weight_var = soft_weight.cuda()
            loss = -((prob[:, :self.num_classes].sum(1) + soft_weight_var).log().mean())
        else:
            loss = -(prob[:, :self.num_classes].sum(1).log().mean())
        return loss


def train(iteration, proto, source_train_loader, source_train_loader_batch, target_train_loader,
          target_train_loader_batch, model, criterion_classifier_source, criterion_classifier_target,
          criterion_em_target, criterion, optimizer, epoch, epoch_count_dataset, args):
    # if iteration * 32 > 1000:
    #     epoch = epoch + 1
    #     new_epoch_flag = True
    #     return source_train_loader_batch, target_train_loader_batch, epoch, new_epoch_flag, iteration
    head = 1
    categories = args.num_classes
    rl_criterion = RewardCriterion()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_classifier = AverageMeter()
    losses_G = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    model.train()
    new_epoch_flag = False
    end = time.time()
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'source':
            epoch = epoch + 1
            new_epoch_flag = True
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]

    try:
        (input_target, _) = target_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'target':
            epoch = epoch + 1
            new_epoch_flag = True
        target_train_loader_batch = enumerate(target_train_loader)
        (input_target, _) = target_train_loader_batch.__next__()[1]
    data_time.update(time.time() - end)

    target_source_temp = target_source + args.num_classes
    target_source_temp = target_source_temp.cuda()
    target_source_temp_var = torch.autograd.Variable(target_source_temp)  #### labels for target classifier

    target_source = target_source.cuda()
    input_source_var = torch.autograd.Variable(input_source)
    target_source_var = torch.autograd.Variable(target_source)  ######## labels for source classifier.
    ############################################ for source samples
    output_source, features_content_S, content_class_S, action_imgs_all_S, \
    sampled_log_img_all_S, log_all_T = model(input_source_var, train=True)

    alpha = 0.80
    features_proto = normalize(features_content_S, -1)
    for i in range(args.num_classes):
        matric = features_proto[target_source_var == i, :]
        if matric.size(0) > 0:
            if iteration <= 1:
                proto[i, :] = matric.mean(0)
            else:
                proto[i, :] = alpha * proto[i, :] + (1 - alpha) * (matric.mean(0))
    iteration = iteration + 1

    loss_task_s_Cs = criterion(output_source[:, :args.num_classes], target_source_var)
    loss_task_s_Ct = criterion(output_source[:, args.num_classes:], target_source_var)

    loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
    loss_category_st_G = 0.5 * criterion(output_source, target_source_var) + 0.5 * criterion(output_source,  ## loss_G
                                                                                             target_source_temp_var)

    input_target_var = torch.autograd.Variable(input_target)
    output_target, features_content_T, content_class_T, action_imgs_all_T, \
    sampled_log_img_all_T, log_all_T = model(input_target_var, train=True)

    # loss_discrim_target = TargetDiscrimLoss(num_classes=categories)(output_target)
    loss_content = nn.BCEWithLogitsLoss()(torch.cat([content_class_S, content_class_T], 1),
                                          torch.cat([torch.ones(content_class_S.size()).cuda(),
                                                     torch.zeros(content_class_S.size()).cuda()], 1))

    loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)
    loss_domain_st_G = 0.5 * criterion_classifier_target(output_target) + 0.5 * criterion_classifier_source(
        output_target)  # loss_G
    loss_target_em = criterion_em_target(output_target)  # # loss_G

    b1 = F.softmax(output_target[:, categories:], -1)
    _, label_target = torch.max(b1, -1)
    certainty_t = ((normalize(features_content_T, -1)
                    .mm(normalize(proto.detach()[label_target, :], -1).t())))
    certainty_t = ((certainty_t - certainty_t.min()) / (certainty_t.max() - certainty_t.min())) / 2.
    certainty_t = certainty_t.t().diag()
    certainty_original = certainty_t

    sort_certainty, sort_index = torch.sort(certainty_original, descending=True, dim=0)
    mask_index = torch.nn.functional.gumbel_softmax(sort_certainty, tau=1, hard=True)  # .argmax()
    temp = [i + 1 for i in range(0, sort_certainty.size(0))]
    index_space = torch.LongTensor(temp).cuda()
    index = (index_space * mask_index).sum(-1)
    mask = torch.cat((torch.ones(int(index.item())), torch.zeros(sort_certainty.size(0) - int(index.item())))).cuda()
    mask[mask_index.argmax().item()] -= 1
    mask = mask_index + mask
    sort_certainty = (mask * sort_certainty)
    _, sort_index2 = torch.sort(sort_index, dim=0)
    certainty_original = sort_certainty.gather(0, sort_index2)
    certainty_margin = certainty_original



    if 1:
        with torch.no_grad():
            reward, _ = eval_func(features_content_S, features_content_S, target_source_var, target_source_var)
            reward2, _ = eval_func(features_content_T, features_content_S, label_target, target_source_var)
            reward3, _ = eval_func(features_content_S, features_content_T, target_source_var, label_target)
            reward_source = (certainty_margin.cpu().squeeze()) * (reward3)
            reward_target = (certainty_margin.cpu().squeeze()) * (reward2)
        text_rl1 = 0.
        text_rl1_t = 0.
        for i in range(head):
            text_rl1 += rl_criterion(sampled_log_img_all_S[:, i, :], action_imgs_all_S[:, i, :],
                                     torch.tensor(reward_source).cuda(), certainty_margin)
            text_rl1_t += rl_criterion(sampled_log_img_all_T[:, i, :], action_imgs_all_T[:, i, :],
                                       torch.tensor(reward_target).cuda(), certainty_margin)

    triplet_loss = TripletLoss(margin=0.5)(features_content_T, features_content_S, label_target, target_source_var,
                                           certainty_margin) \
                   + TripletLoss(margin=0.5)(features_content_S, features_content_T, target_source_var, label_target,
                                             certainty_margin)

    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
    if args.flag == 'no_em':
        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2  ### used to classifier
        loss_G = loss_category_st_G + lam * loss_domain_st_G  ### used to feature extractor

    elif args.flag == 'symnet':  #
        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2  ### used to classifier
        loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em)  ### used to feature extractor

    elif args.flag == 'new':
        loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2 \
                          + 10*(triplet_loss + text_rl1 + text_rl1_t + loss_content)
        loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em)

    else:
        raise ValueError('unrecognized flag:', args.flag)

    # mesure accuracy and record loss
    prec1_source, _ = accuracy(output_source.data[:, :args.num_classes], target_source, topk=(1, 5))
    prec1_target, _ = accuracy(output_source.data[:, args.num_classes:], target_source, topk=(1, 5))
    losses_classifier.update(loss_classifier.item(), input_source.size(0))
    losses_G.update(loss_G.item(), input_source.size(0))
    top1_source.update(prec1_source[0], input_source.size(0))
    top1_target.update(prec1_target[0], input_source.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss_classifier.backward(retain_graph=True)
    temp_grad = []
    for param in model.parameters():
        if param.grad is None:
            continue
        temp_grad.append(param.grad.data.clone())
    grad_for_classifier = temp_grad

    optimizer.zero_grad()
    loss_G.backward()
    temp_grad = []
    for param in model.parameters():
        if param.grad is None:
            continue
        temp_grad.append(param.grad.data.clone())
    grad_for_featureExtractor = temp_grad

    count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()
        if count < 159:  ########### the feautre extractor of the ResNet-50
            temp_grad = temp_grad + grad_for_featureExtractor[count]
        else:
            temp_grad = temp_grad + grad_for_classifier[count] + grad_for_featureExtractor[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    # print(count)
    optimizer.step()
    batch_time.update(time.time() - end)
    end = time.time()
    if (epoch + 1) % args.print_freq == 0 or epoch == 0:
        print('Train: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss@C {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
              'Loss@G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
              'top1S {top1S.val:.3f} ({top1S.avg:.3f})\t'
              'top1T {top1T.val:.3f} ({top1T.avg:.3f})'.format(
            epoch, args.epochs, batch_time=batch_time,
            data_time=data_time, loss_c=losses_classifier, loss_g=losses_G, top1S=top1_source, top1T=top1_target))
        if new_epoch_flag:
            log = open(os.path.join(args.log, 'log.txt'), 'a')
            log.write("\n")
            log.write("Train:epoch: %d, loss@min: %4f, loss@max: %4f, Top1S acc: %3f, Top1T acc: %3f" % (
                epoch, losses_classifier.avg, losses_G.avg, top1_source.avg, top1_target.avg))
            log.close()

    return source_train_loader_batch, target_train_loader_batch, epoch, new_epoch_flag, iteration


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses_source = AverageMeter()
    losses_target = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _), in enumerate(val_loader):  # , _
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        with torch.no_grad():
            output, features_content, content_class, action_imgs_all, \
            sampled_log_img_all, log_all = model(input_var, train=False)
        loss_source = criterion(output[:, :args.num_classes], target_var)
        loss_target = criterion(output[:, args.num_classes:], target_var)
        # measure accuracy and record loss
        prec1_source, _ = accuracy(output.data[:, :args.num_classes], target, topk=(1, 5))
        prec1_target, _ = accuracy(output.data[:, args.num_classes:], target, topk=(1, 5))

        losses_source.update(loss_source.item(), input.size(0))
        losses_target.update(loss_target.item(), input.size(0))

        top1_source.update(prec1_source[0], input.size(0))
        top1_target.update(prec1_target[0], input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'LS {lossS.val:.4f} ({lossS.avg:.4f})\t'
                  'LT {lossT.val:.4f} ({lossT.avg:.4f})\t'
                  'top1S {top1S.val:.3f} ({top1S.avg:.3f})\t'
                  'top1T {top1T.val:.3f} ({top1T.avg:.3f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time, lossS=losses_source, lossT=losses_target,
                top1S=top1_source, top1T=top1_target))

    print(' * Top1@S {top1S.avg:.3f} Top1@T {top1T.avg:.3f}'
          .format(top1S=top1_source, top1T=top1_target))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n")
    log.write("                                    Test:epoch: %d, LS: %4f, LT: %4f, Top1S: %3f, Top1T: %3f" % \
              (epoch, losses_source.avg, losses_target.avg, top1_source.avg, top1_target.avg))
    log.close()
    return max(top1_source.avg, top1_target.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    ## annealing strategy 1
    # epoch_total = int(args.epochs / args.test_freq)
    # epoch = int((epoch + 1) / args.test_freq)
    lr = args.lr / pow((1 + 10 * epoch / args.epochs), 0.75)
    lr_pretrain = args.lr * 0.1 / pow((1 + 10 * epoch / args.epochs),
                                      0.75)  # 0.001 / pow((1 + 10 * epoch / epoch_total), 0.75)
    ## annealing strategy 2
    # exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    # lr = args.lr * (args.gamma ** exp)
    # lr_pretrain = lr * 0.1 #1e-3
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = lr_pretrain
        else:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



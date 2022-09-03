import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import ipdb
import torch.nn.functional as F
from pytorch_revgrad import RevGrad
import torchvision.ops.roi_pool as roi_pool

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def sliding_window(batchsize, stepSize, windowSize):
    boxes_all = []
    for j in range(batchsize):
      boxes = []
      for i in range(2):
          for y in range(0, 224 - stepSize[i], stepSize[i]):  # 224
            for x in range(0, 224 - stepSize[i], stepSize[i]):
              boxes.append([x, y, x + windowSize[i], y + windowSize[i]])
      boxes = torch.tensor(boxes).cuda().float()
      boxes_all.append(boxes)
    return boxes_all


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def sample_softmax(logits, train, temperature=0.6, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    b, h, d = logits.size()
    logits = logits.view(-1, d)
    y = torch.softmax(logits, dim=-1)
    if torch.isnan(y).any():
        y = torch.nan_to_num(y)
    y = y.clamp(min=1e-20, max=1e+20)
    if train == True:
        Cdis = torch.distributions.Categorical(y)
        ind = Cdis.sample()
        prob = (Cdis.log_prob(ind)).exp()
    else:
        prob, ind = y.max(dim=-1)
    return ind.view(b, h, -1), prob.view(b, h, -1)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=65*2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.ATT = 1
        self.head = 1
        embed_size = 2048
        feat_size = 2048
        self.aspace = 53
        self.headap = 8

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.rl_rnn = nn.GRUCell(feat_size// self.headap, feat_size// self.headap)
        self.linear_rl_image_mus = nn.Linear(feat_size // self.headap, self.aspace * self.head)

        self.linear_last = nn.Linear(feat_size, num_classes)
        self.dis_linear = nn.Linear(feat_size, feat_size // 2)
        self.dis_classify = nn.Linear(feat_size // 2, 1)

        self.bn = nn.BatchNorm1d(embed_size)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        # aspace = 74 #49   53
        embed_size = 2048
        feat_size = 2048

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # content = self.avgpool(x).view(x.size(0), -1)
        content = x

        action_img_all = []
        sampled_log_img = []
        log_all = []
        img_mu_all = []
        img_att_all = []
        hx_image = torch.zeros([x.size(0), feat_size//self.headap]).cuda()
        feature_map = content.view(x.size(0), self.headap, embed_size//self.headap, 49).permute(0,3,1,2)
        hidden_all = []
        if 1:
            for i in range(feature_map.size(1)):
                if i <= feature_map.size(1) - 1:
                    img = feature_map[:, i, :]
                feat_hx = []
                hiddens = []
                for j in range(self.headap):
                    img_rnn = img[:, j, :]
                    hx_image = self.rl_rnn(img_rnn, hx_image)
                    feat_hx.append(hx_image.unsqueeze(1))
                    hiddens.append(hx_image.unsqueeze(1))
                hidden_all.append(torch.cat(hiddens, dim=1).view(x.size(0), embed_size))
                hx_image2 = torch.cat(feat_hx, dim=1)
                img_mu = self.linear_rl_image_mus(hx_image2).sum(1)
                img_mu = img_mu.view(-1, self.head, self.aspace)
                ind, prob = sample_softmax(img_mu, train)
                prob = prob.squeeze()
                action_img = ind.float()
                action_img_all += [action_img]
                sampled_log_img += [torch.log(prob).squeeze().unsqueeze(-1)]
                img_mu_all += [img_mu]
                img_att = action_img.view(-1, self.head, 1)
                img_att_all.append(img_att)
                img = img.view(img.size(0), -1)
            sampled_log_img_all = torch.stack(sampled_log_img, 1).squeeze()
            sampled_log_img_all = sampled_log_img_all.view(feature_map.size(0), self.head, -1)
            action_imgs_all = torch.stack(action_img_all, 1).view(feature_map.size(0), self.head, -1)
            img_att_all = torch.stack(img_att_all, 0).mean(2).squeeze().t()
            img_att_all = F.softmax(img_att_all / 0.05, -1)
            hidden_all = torch.stack(hidden_all, 0).transpose(1, 0)
            features = (hidden_all * img_att_all.unsqueeze(-1)).sum(1)
            features_content = features
        else:
            features_content = content.mean(3).mean(2)

        features_content = self.bn((features_content+content.mean(3).mean(2))) #+ content  # +content.mean(3).mean(2)
        features_classifier = self.linear_last(self.dp1(features_content))
        content_feat = F.relu(self.dis_linear(RevGrad(alpha=0.5)(features_content)))  # RevGrad(alpha=0.5)
        content_class = self.dis_classify(self.dp2(content_feat))

        return features_classifier, features_content, content_class, action_imgs_all, sampled_log_img_all, log_all


def resnet18(args, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)
    model.fc.weight.data.normal_(0.0, 0.02)
    model.fc.bias.data.normal_(0)
    return model


def resnet34(args, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        print('Load ImageNet pre-trained resnet model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)

    return model


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        if args.pretrained_checkpoint:  ################### use self-pretrained model
            # modify the structure of the model.
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
            init_dict = model.state_dict()
            pretrained_dict_temp = torch.load(args.pretrained_checkpoint)['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict_temp.items()}
            temp = init_dict['fc.weight'].clone()
            temp[:args.num_classes, :] = pretrained_dict['fc.weight'].clone()
            pretrained_dict['fc.weight'] = temp.clone()
            temp = init_dict['fc.bias'].clone()
            temp[:args.num_classes] = pretrained_dict['fc.bias'].clone()
            pretrained_dict['fc.bias'] = temp.clone()
            model.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                                  strict=False)  ########## use imagenet pretrained model
            # modify the structure of the model.
            # num_of_feature_map = model.fc.in_features
            # model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
    return model


def resnet101(args, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if args.pretrained:
        if args.pretrained_checkpoint:  ################### use self-pretrained model
            # modify the structure of the model.
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
            init_dict = model.state_dict()
            pretrained_dict_temp = torch.load(args.pretrained_checkpoint)['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict_temp.items()}
            temp = init_dict['fc.weight'].clone()
            temp[:args.num_classes, :] = pretrained_dict['fc.weight'].clone()
            pretrained_dict['fc.weight'] = temp.clone()
            temp = init_dict['fc.bias'].clone()
            temp[:args.num_classes] = pretrained_dict['fc.bias'].clone()
            pretrained_dict['fc.bias'] = temp.clone()
            model.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),
                                  strict=False)
    return model


def resnet152(args, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)

    return model


def resnet(args, **kwargs):  ################ Only support ResNet-50 in this simple code.
    print("==> creating model '{}' ".format(args.arch))
    if args.arch == 'resnet18':
        return resnet18(args)
    elif args.arch == 'resnet34':
        return resnet34(args)
    elif args.arch == 'resnet50':
        return resnet50(args)
    elif args.arch == 'resnet101':
        return resnet101(args)
    elif args.arch == 'resnet152':
        return resnet152(args)
    else:
        raise ValueError('Unrecognized model architecture', args.arch)
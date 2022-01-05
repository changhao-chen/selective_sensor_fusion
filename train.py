import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import argparse
import time
import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
from models import FlowNet, RecFeat, PoseRegressor, RecImu, Fc_Flownet, Hard_Mask, Soft_Mask
from utils import save_path_formatter, mat2euler
from logger import AverageMeter
from itertools import chain
import torch.nn.functional as F
from data_loader import KITTI_Loader

parser = argparse.ArgumentParser(description='Selective Sensor Fusion on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')

best_error = -1
n_iter = 0


def main():

    global best_error, n_iter

    args = parser.parse_args()

    n_save_model = 1
    degradation_mode = 0
    fusion_mode = 3
    # 0: vision only 1: direct 2: soft 3: hard

    # set saving path
    abs_path = ''
    save_path = save_path_formatter(args, parser)
    args.save_path = abs_path + 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    if not os.path.exists(args.save_path+'/imgs/'):
        os.makedirs(args.save_path+'/imgs/')
    if not os.path.exists(args.save_path+'/models/'):
        os.makedirs(args.save_path+'/models/')
    torch.manual_seed(args.seed)

    # image transform
    normalize = custom_transforms.Normalize(mean=[0, 0, 0],
                                            std=[255, 255, 255])
    normalize2 = custom_transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    input_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize,
        normalize2
    ])

    # Data loading code
    print("=> fetching scenes in '{}'".format(args.data))
    train_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        seed=args.seed,
        train=0,
        sequence_length=args.sequence_length,
        data_degradation=degradation_mode, data_random=True
    )

    val_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        seed=args.seed,
        train=1,
        sequence_length=args.sequence_length,
        data_degradation=degradation_mode, data_random=True
    )

    test_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        seed=args.seed,
        train=2,
        sequence_length=args.sequence_length,
        data_degradation=degradation_mode, data_random=False
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create pose model
    print("=> creating pose model")

    feature_dim = 256

    if fusion_mode == 0:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim*2).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 1:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 2:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Soft_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 3:
        feature_ext = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    pose_net.init_weights()

    flownet_model_path = abs_path + '../../../pretrain/flownets_EPE1.951.pth'
    pretrained_flownet = True
    if pretrained_flownet:
        weights = torch.load(flownet_model_path)
        model_dict = feature_ext.state_dict()
        update_dict = {k: v for k, v in weights['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        feature_ext.load_state_dict(model_dict)
        print('restrore depth model from ' + flownet_model_path)

    cudnn.benchmark = True
    feature_ext = torch.nn.DataParallel(feature_ext)
    rec_feat = torch.nn.DataParallel(rec_feat)
    pose_net = torch.nn.DataParallel(pose_net)
    rec_imu = torch.nn.DataParallel(rec_imu)
    fc_flownet = torch.nn.DataParallel(fc_flownet)
    selectfusion = torch.nn.DataParallel(selectfusion)

    print('=> setting adam solver')

    parameters = chain(rec_feat.parameters(), rec_imu.parameters(), fc_flownet.parameters(), pose_net.parameters(),
                       selectfusion.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'train_pose', 'train_euler', 'validation_loss', 'val_pose', 'val_euler'])

    # start training loop
    print('=> training pose model')

    best_val = 100

    best_tra = 10000.0
    best_ori = 10000.0

    for epoch in range(args.epochs):

        # train for one epoch
        train_loss, pose_loss, euler_loss, temp = train(args, train_loader, feature_ext, rec_feat, rec_imu, pose_net,
                                                        fc_flownet, selectfusion, optimizer, epoch, fusion_mode)

        temp = 0.5

        # evaluate on validation set
        val_loss, val_pose_loss, val_euler_loss =\
            validate(args, val_loader, feature_ext, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, temp, epoch,
                     fusion_mode)

        # evaluate on validation set
        test(args, test_loader, feature_ext, rec_feat, rec_imu, pose_net,
             fc_flownet, selectfusion, temp, epoch, fusion_mode)

        if val_pose_loss < best_tra:
            best_tra = val_pose_loss

        if val_euler_loss < best_ori:
            best_ori = val_euler_loss

        print('Best: {}, Best Translation {:.5} Best Orientation {:.5}'
              .format(epoch + 1, best_tra, best_ori))

        # save checkpoint
        if (epoch % n_save_model == 0) or (val_loss < best_val):

            best_val = val_loss

            fn = args.save_path + '/models/rec_' + str(epoch) + '.pth'
            torch.save(rec_feat.module.state_dict(), fn)

            fn = args.save_path + '/models/pose_' + str(epoch) + '.pth'
            torch.save(pose_net.module.state_dict(), fn)

            fn = args.save_path + '/models/fc_flownet_' + str(epoch) + '.pth'
            torch.save(fc_flownet.module.state_dict(), fn)

            fn = args.save_path + '/models/rec_imu_' + str(epoch) + '.pth'
            torch.save(rec_imu.module.state_dict(), fn)

            fn = args.save_path + '/models/selectfusion_' + str(epoch) + '.pth'
            torch.save(selectfusion.module.state_dict(), fn)
            print('Model has been saved')

        with open(args.save_path / args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, pose_loss, euler_loss, val_loss, val_pose_loss, val_euler_loss])


def train(args, train_loader, feature_ext, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, optimizer, epoch,
          fusion_mode):

    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()

    epoch_size = args.epoch_size

    # switch to train mode
    feature_ext.eval()
    rec_feat.train()
    rec_imu.train()
    pose_net.train()
    fc_flownet.train()
    selectfusion.train()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    temp_min = 0.5
    ANNEAL_RATE = 0.0006
    temp = 1.0

    for i, (imgs, imus, poses) in enumerate(train_loader):

        if len(imgs[0]) != args.batch_size:
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        rec_feat.module.hidden = rec_feat.module.init_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs)-1):

            tgt_img = imgs[j+1]
            ref_img = imgs[j]

            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            rec_imu.module.hidden = rec_imu.module.init_hidden()

            with torch.no_grad():

                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)

            feature_vision = fc_flownet(raw_feature_vision)

            if fusion_mode == 0:

                feature_weighted = feature_vision

            else:

                # extract imu features
                feature_imu = rec_imu(imu_var)

                # concatenate visual and imu features
                feature = torch.cat([feature_vision, feature_imu], 2)

                if fusion_mode == 1:

                    feature_weighted = feature

                else:

                    if fusion_mode == 2:
                        mask = selectfusion(feature)

                    else:
                        mask = selectfusion(feature, temp)

                    feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask


            # recurrent features
            feature_new = rec_feat(feature_weighted)

            # pose net
            pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_loss += loss.item()
        aver_pose_loss += pose_loss.item()
        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Train: Epoch [{}/{}] Step [{}/{}]: Time {} Data {} Loss {:.5} '
                  'Pose {:.5} Euler {:.5}'.
                  format(epoch + 1, args.epochs, i + 1, epoch_size,
                         batch_time, data_time, loss.item(), pose_loss.item(), euler_loss.item()))

        # decrease hard mask temperature
        if i % 10 == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)

        if i >= epoch_size - 1:
            break

        n_iter += 1

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n

    print('Train: {}, Average_Loss {:.5} pose_loss {:.5} euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return aver_loss, aver_pose_loss, aver_euler_loss, temp


def validate(args, val_loader, feature_ext, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, temp, epoch,
             fusion_mode):

    batch_time = AverageMeter()

    # switch to evaluate mode
    feature_ext.eval()
    rec_feat.eval()
    pose_net.eval()
    rec_imu.eval()
    fc_flownet.eval()
    selectfusion.eval()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    for i, (imgs, imus, poses) in enumerate(val_loader):

        if len(imgs[0]) != args.batch_size:
            continue

        rec_feat.module.hidden = rec_feat.module.init_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs) - 1):

            tgt_img = imgs[j + 1]
            ref_img = imgs[j]
            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            with torch.no_grad():

                rec_imu.module.hidden = rec_imu.module.init_hidden()

                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)

                feature_vision = fc_flownet(raw_feature_vision)

                if fusion_mode == 0:

                    feature_weighted = feature_vision

                else:

                    # extract imu features
                    feature_imu = rec_imu(imu_var)

                    # concatenate visual and imu features
                    feature = torch.cat([feature_vision, feature_imu], 2)

                    if fusion_mode == 1:

                        feature_weighted = feature

                    else:

                        if fusion_mode == 2:
                            mask = selectfusion(feature)

                        else:
                            mask = selectfusion(feature, temp)

                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                # recurrent features
                feature_new = rec_feat(feature_weighted)

                pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_pose_loss += pose_loss.item()
        aver_loss += loss.item()

        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Val: Epoch [{}/{}] Step [{}/{}]: Loss {:.5} '
                  'Pose {:.5} Euler {:.5}'.
                  format(epoch + 1, args.epochs, i + 1, len(val_loader), loss.item(), pose_loss.item(),
                         euler_loss.item()))

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n
    print('Val: {}, Average_Loss {:.5} Pose_loss {:.5} Euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return aver_loss, aver_pose_loss, aver_euler_loss


def test(args, test_loader, feature_ext, rec_feat, rec_imu, pose_net,
         fc_flownet, selectfusion, temp, epoch, fusion_mode):

    batch_time = AverageMeter()

    # switch to evaluate mode
    feature_ext.eval()
    rec_feat.eval()
    pose_net.eval()
    rec_imu.eval()
    fc_flownet.eval()
    selectfusion.eval()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    for i, (imgs, imus, poses) in enumerate(test_loader):

        if i == 0:
            k = 5
        if i == 1:
            k = 7
        if i == 2:
            k = 10

        result = []
        truth_pose = []
        truth_euler = []

        rec_feat.module.hidden = rec_feat.module.init_test_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs) - 1):

            tgt_img = imgs[j + 1]
            ref_img = imgs[j]
            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            with torch.no_grad():

                rec_imu.module.hidden = rec_imu.module.init_test_hidden()

                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)

                feature_vision = fc_flownet(raw_feature_vision)

                if fusion_mode == 0:

                    feature_weighted = feature_vision

                else:

                    # extract imu features
                    feature_imu = rec_imu(imu_var)

                    # concatenate visual and imu features
                    feature = torch.cat([feature_vision, feature_imu], 2)

                    if fusion_mode == 1:

                        feature_weighted = feature

                    else:

                        if fusion_mode == 2:
                            mask = selectfusion(feature)

                        else:
                            mask = selectfusion(feature, temp)

                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                # recurrent features
                feature_new = rec_feat(feature_weighted)

                pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            if len(result) == 0:
                result = np.copy(pose.cpu().detach().numpy())
            else:
                result = np.concatenate((result, pose.cpu().detach().numpy()), axis=0)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

            if len(truth_pose) == 0:
                truth_pose = np.copy(pose_truth.cpu().detach().numpy())
            else:
                truth_pose = np.concatenate((truth_pose, pose_truth.cpu().detach().numpy()), axis=0)

            if len(truth_euler) == 0:
                truth_euler = np.copy(euler.cpu().detach().numpy())
            else:
                truth_euler = np.concatenate((truth_euler, euler.cpu().detach().numpy()), axis=0)

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_pose_loss += pose_loss.item()
        aver_loss += loss.item()

        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test Seq{}: Epoch [{}/{}] Step [{}/{}]: Loss {:.5} '
              'Pose {:.5} Euler {:.5}'.
              format(k, epoch + 1, args.epochs, i + 1, len(test_loader), loss.item(), pose_loss.item(),
                     euler_loss.item()))

        file_name = 'results/result_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, result, delimiter=',')

        file_name = 'results/truth_pose_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, truth_pose, delimiter=',')

        file_name = 'results/truth_euler_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, truth_euler, delimiter=',')

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n
    print('Test Average: {}, Average_Loss {:.5} Pose_loss {:.5} Euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return


def compute_trans_pose(ref_pose, tgt_pose):

    tmp_pose = np.copy(tgt_pose)

    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]
    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose

    return trans_pose


if __name__ == '__main__':
    main()

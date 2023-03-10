import os
import faiss
import torch 
import shutil 
import numpy as np

from tqdm import tqdm 
from model.SFUniDA import SFUniDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

best_score = 0.0
best_coeff = 1.0

@torch.no_grad()
def obtain_global_pseudo_labels(args, model, dataloader, epoch_idx=0.0):
    model.eval()

    pred_cls_bank = [] 
    gt_label_bank = []
    embed_feat_bank = []
    class_list = args.target_class_list
    
    args.logger.info("Generating one-vs-all global clustering pseudo labels...")
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        embed_feat, pred_cls = model(imgs_test, apply_softmax=True)
        pred_cls_bank.append(pred_cls)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(imgs_label.cuda())
    
    pred_cls_bank = torch.cat(pred_cls_bank, dim=0) #[N, C]
    gt_label_bank = torch.cat(gt_label_bank, dim=0) #[N]
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) #[N, D]
    embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    
    global best_score
    global best_coeff
    # At the first epoch, we need to determine the number of categories in target domain, i.e., the C_t in our paper.
    # Here, we utilize the Silhouette metric to realize this goal.
    if epoch_idx == 0.0:
        embed_feat_bank_cpu = embed_feat_bank.cpu().numpy()
        
        if args.dataset == "VisDA" or args.dataset == "DomainNet":
            # np.random.seed(2021)
            data_size = embed_feat_bank_cpu.shape[0]
            sample_idxs = np.random.choice(data_size, data_size//3, replace=False)
            embed_feat_bank_cpu = embed_feat_bank_cpu[sample_idxs, :]
            
        embed_feat_bank_cpu = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embed_feat_bank_cpu)
        coeff_list = [0.25, 0.50, 1, 2, 3]
        
        for coeff in coeff_list:
            KK = max(int(args.class_num * coeff), 2)
            kmeans = KMeans(n_clusters=KK, random_state=0).fit(embed_feat_bank_cpu)
            cluster_labels = kmeans.labels_
            sil_score = silhouette_score(embed_feat_bank_cpu, cluster_labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_coeff = coeff
    
    KK = int(args.class_num * best_coeff)
    
    data_num = pred_cls_bank.shape[0]
    pos_topk_num = int(data_num / args.class_num / best_coeff)
    sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True)
    pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :].t() #[C, pos_topk_num]
    neg_topk_idxs = sorted_pred_cls_idxs[pos_topk_num:, :].t() #[C, neg_topk_num]
    
    pos_topk_idxs = pos_topk_idxs.unsqueeze(2).expand([-1, -1, args.embed_feat_dim]) #[C, pos_topk_num, D]
    neg_topk_idxs = neg_topk_idxs.unsqueeze(2).expand([-1, -1, args.embed_feat_dim]) #[C, neg_topk_num, D]
    
    embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([args.class_num, -1, -1]) #[C, N, D]
    pos_feat_sample = torch.gather(embed_feat_bank_expand, 1, pos_topk_idxs)
        
    pos_cls_prior = torch.mean(sorted_pred_cls[:(pos_topk_num), :], dim=0, keepdim=True).t() * (1.0 - args.rho) + args.rho
    
    args.logger.info("POS_CLS_PRIOR:\t" + "\t".join(["{:.3f}".format(item) for item in pos_cls_prior.cpu().squeeze().numpy()]))
    
    pos_feat_proto = torch.mean(pos_feat_sample, dim=1, keepdim=True) #[C, 1, D]
    pos_feat_proto = pos_feat_proto / torch.norm(pos_feat_proto, p=2, dim=-1, keepdim=True)

    faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, KK, niter=100, verbose=False, min_points_per_centroid=1, gpu=False)
    
    feat_proto_pos_simi = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    feat_proto_max_simi = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    feat_proto_max_idxs = torch.zeros((data_num, args.class_num)).cuda() #[N, C]
    
    # One-vs-all class pseudo-labeling
    for cls_idx in range(args.class_num):
        neg_feat_cls_sample_np = torch.gather(embed_feat_bank, 0, neg_topk_idxs[cls_idx, :]).cpu().numpy()
        faiss_kmeans.train(neg_feat_cls_sample_np)
        cls_neg_feat_proto = torch.from_numpy(faiss_kmeans.centroids).cuda()
        cls_neg_feat_proto = cls_neg_feat_proto / torch.norm(cls_neg_feat_proto, p=2, dim=-1, keepdim=True)#[K, D]
        cls_pos_feat_proto = pos_feat_proto[cls_idx, :] #[1, D]
        
        cls_pos_feat_proto_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_pos_feat_proto) #[N, 1]
        cls_neg_feat_proto_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_neg_feat_proto) #[N, K]
        cls_pos_feat_proto_simi = cls_pos_feat_proto_simi * pos_cls_prior[cls_idx] #[N, 1]
        
        cls_feat_proto_simi = torch.cat([cls_pos_feat_proto_simi, cls_neg_feat_proto_simi], dim=1) #[N, 1+K]
        
        feat_proto_pos_simi[:, cls_idx] = cls_feat_proto_simi[:, 0]
        maxsimi, maxidxs = torch.max(cls_feat_proto_simi, dim=-1)
        feat_proto_max_simi[:, cls_idx] = maxsimi
        feat_proto_max_idxs[:, cls_idx] = maxidxs
    
    # we use this psd_label_prior_simi to control the hard pseudo label either one-hot or unifrom distribution. 
    psd_label_prior_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, pos_feat_proto.squeeze(1))
    psd_label_prior_idxs = torch.max(psd_label_prior_simi, dim=-1, keepdim=True)[1] #[N] ~ (0, class_num-1)
    psd_label_prior = torch.zeros_like(psd_label_prior_simi).scatter(1, psd_label_prior_idxs, 1.0) # one_hot prior #[N, C]
    
    hard_psd_label_bank = feat_proto_max_idxs # [N, C] ~ (0, K)
    hard_psd_label_bank = (hard_psd_label_bank == 0).float()
    hard_psd_label_bank = hard_psd_label_bank * psd_label_prior #[N, C]
    
    hard_label = torch.argmax(hard_psd_label_bank, dim=-1) #[N]
    hard_label_unk = torch.sum(hard_psd_label_bank, dim=-1) 
    hard_label_unk = (hard_label_unk == 0)
    hard_label[hard_label_unk] = args.class_num
    
    hard_psd_label_bank[hard_label_unk, :] += 1.0
    hard_psd_label_bank = hard_psd_label_bank / (torch.sum(hard_psd_label_bank, dim=-1, keepdim=True) + 1e-4)
    
    hard_psd_label_bank = hard_psd_label_bank.cuda()
    
    per_class_num = np.zeros((len(class_list)))
    pre_class_num = np.zeros_like(per_class_num)
    per_class_correct = np.zeros_like(per_class_num)
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_bank == label)[0]
        correct_idx = torch.where(hard_label[label_idx] == label)[0]
        pre_class_num[i] = float(len(torch.where(hard_label == label)[0]))
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    per_class_acc = per_class_correct / (per_class_num + 1e-5)
    
    args.logger.info("PSD AVG ACC:\t" + "{:.3f}".format(np.mean(per_class_acc)))
    args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))
    args.logger.info("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_num]))
    args.logger.info("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pre_class_num]))
    args.logger.info("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_correct]))
    
    return hard_psd_label_bank, pred_cls_bank, embed_feat_bank


def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    
    model.eval()
    hard_psd_label_bank, pred_cls_bank, embed_feat_bank = obtain_global_pseudo_labels(args, model, test_dataloader,epoch_idx) 
    model.train()
    
    local_KNN = args.local_K
    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    knn_pred_loss_stack = []
    
    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    
    for imgs_train, _, _, imgs_idx in tqdm(train_dataloader, ncols=60):
        
        iter_idx += 1
        imgs_idx = imgs_idx.cuda()
        imgs_train = imgs_train.cuda()
        
        hard_psd_label = hard_psd_label_bank[imgs_idx] #[B, C]
        
        embed_feat, pred_cls = model(imgs_train, apply_softmax=True)
        
        psd_pred_loss = torch.sum(-hard_psd_label * torch.log(pred_cls + 1e-5), dim=-1).mean()
        
        with torch.no_grad():
            embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=-1, keepdim=True)
            feat_dist = torch.einsum("bd, nd -> bn", embed_feat, embed_feat_bank) #[B, N]
            nn_feat_idx = torch.topk(feat_dist, k=local_KNN+1, dim=-1, largest=True)[-1] #[B, local_KNN+1]
            nn_feat_idx = nn_feat_idx[:, 1:] #[B, local_KNN]
            nn_pred_cls = torch.mean(pred_cls_bank[nn_feat_idx], dim=1) #[B, C]
            # update the pred_cls and embed_feat bank 
            pred_cls_bank[imgs_idx] = pred_cls
            embed_feat_bank[imgs_idx] = embed_feat
        
        knn_pred_loss = torch.sum(-nn_pred_cls * torch.log(pred_cls + 1e-5), dim=-1).mean()
        
        loss = args.lam_psd * psd_pred_loss + args.lam_knn * knn_pred_loss
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_pred_loss_stack.append(loss.cpu().item())
        psd_pred_loss_stack.append(psd_pred_loss.cpu().item())
        knn_pred_loss_stack.append(knn_pred_loss.cpu().item())
        
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
    train_loss_dict["psd_pred_loss"] = np.mean(psd_pred_loss_stack)
    train_loss_dict["knn_pred_loss"] = np.mean(knn_pred_loss_stack)
            
    return train_loss_dict
    
@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda() 
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc
    
def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
    
    model = model.cuda()
    save_dir = os.path.join(this_dir, "checkpoints_glc", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False  
        
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)
    
    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="
    
    args.logger.info(notation_str)
    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target
        loss_dict =train(args, model, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx)
        args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
                          train_psd_loss:{:.3f}, train_knn_loss:{:.3f},".format(epoch_idx+1, args.epochs,
                                        loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"], loss_dict["knn_pred_loss"]))
        
        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
        args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))
        
        if args.target_label_type == 'PDA' or args.target_label_type == 'CLDA':
            if knownacc >= best_known_acc:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
                
                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)         
                # torch.save({
                #     "epoch":epoch_idx,
                #     "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
            
                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)         
                # torch.save({
                #     "epoch":epoch_idx,
                #     "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
            
        args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc, best_unknown_acc))
            
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    
    # SET THE CHECKPOINT     
    args.checkpoint = os.path.join("checkpoints_glc", args.dataset, "source_{}".format(args.s_idx),\
                    "source_{}_{}".format(args.source_train_type, args.target_label_type),
                    "latest_source_checkpoint.pth")
    main(args)

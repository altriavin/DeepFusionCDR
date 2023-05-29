import os
import torch
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from dataloader import *
import pandas as pd
from tensorboardX import SummaryWriter


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    for step, x in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            h = model.forward_cluster(x)
        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())
    feature_vector = np.array(feature_vector)
    return feature_vector


def train():
    loss_epoch = 0
    for step, x in enumerate(DL):
        optimizer.zero_grad()
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        batch = x_i.shape[0]
        criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
        criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)
        loss_instance = criterion_instance(z_i, z_j)+criterion_instance(z_j, z_i)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    cluster_number = 2

    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    model_path = './save/' + args.cancer_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = SummaryWriter(log_dir="./log")

    DL=get_feature(args.cancer_type, args.batch_size, True)

    ae = ae.AE()
    model = network.Network(ae, args.feature_dim, cluster_number)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_device = device
    
    loss=[]
    for epoch in range(args.start_epoch, args.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        loss.append(loss_epoch)
        logger.add_scalar("train loss", loss_epoch)
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
    save_model(model_path, model, optimizer, args.epochs)

    dataloader = get_feature(args.batch_size, False)
    
    model = network.Network(ae, args.feature_dim, cluster_number)
    model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    h = inference(dataloader, model, device)

    fea_out_file = './results/GDSC.csv'
    fea = pd.DataFrame(data=h)
    fea.to_csv(fea_out_file, header=False, index=False, sep='\t')

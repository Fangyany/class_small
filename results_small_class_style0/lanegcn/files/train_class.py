import os

import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

import torch
from torch.utils.data import Sampler, DataLoader
import torch.nn as nn
import torch.optim as optim

from utils import Logger, load_pretrain
from lanegcn import get_fake_traj_rel, get_pred_traj_rel, TrajectoryDiscriminator, Loss, get_model
from loss import gan_d_loss, gan_g_loss

torch.cuda.set_device(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="model", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)


def main():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



    # Import all settings for experiment.
    args = parser.parse_args()
    config, Dataset, collate_fn, generator, loss_l2, post_process, opt = get_model()

    discriminator = TrajectoryDiscriminator(config)
    optimizer_g = optim.Adam(generator.parameters(), lr=5e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=5e-4)
    generator.to("cuda")
    discriminator.to("cuda")


    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(generator, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset('./dataset/preprocess/train_crs_dist6_angle90.p', config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, generator, discriminator, loss_l2, post_process, optimizer_g, optimizer_d)







def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, generator, discriminator, loss_l2, post_process, optimizer_g, optimizer_d):
    # train_loader.sampler.set_epoch(int(epoch))
    generator.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (config["batch_size"]))

    
     
    d_loss_fn = gan_d_loss
    g_loss_fn = gan_g_loss
        
    

    d_steps_left = 2
    g_steps_left = 1


    trajectory_style = torch.load('./style0.pth').cuda()


    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(train_loader):
        epoch += epoch_per_batch
        data = dict(data)
        out_rel, out = generator(data)

        if g_steps_left > 0:
            losses_g = generator_step(data, out_rel, out, loss_l2, g_loss_fn, discriminator, optimizer_g)
            g_steps_left -= 1 
        elif d_steps_left > 0:
            losses_d = discriminator_step(data, trajectory_style, out_rel, discriminator, d_loss_fn, optimizer_d)
            d_steps_left -= 1

        d_steps_left = 2
        g_steps_left = 1

        
        loss_out = loss_l2(out, data)
        post_out = post_process(out, data)
        post_process.append(metrics, loss_out, post_out)



        # lr = optimizer_g.step(epoch)
        lr = optimizer_g.step()




        num_iters = int(np.round(epoch * num_batches))
        if num_iters % save_iters == 0 or epoch >= config["num_epochs"]:
            save_ckpt(generator, optimizer_g, config["save_dir"], "generator", epoch)
            save_ckpt(discriminator, optimizer_d, config["save_dir"], "discriminator", epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            # metrics = sync(metrics)
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()




def save_ckpt(net, opt, save_dir, model_name, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = model_name + "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.state_dict()},
        os.path.join(save_dir, save_name),
    )

   

def discriminator_step(data, trajectory_style, out_rel, discriminator, d_loss_fn, optimizer_d):

    fake_traj_rel = get_fake_traj_rel(data['traj1'], out_rel['reg'])
    pred_traj_rel = trajectory_style
    scores_real = discriminator(pred_traj_rel)
    scores_fake = discriminator(fake_traj_rel)


    losses = {}
    loss = torch.zeros(1)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()

    optimizer_d.step()

    return losses


def generator_step(data, out_rel, out, loss_l2, g_loss_fn, discriminator, optimizer_g):
    
    losses = {}
    loss = torch.zeros(1).cuda()

    loss_out = loss_l2(out, data)
    losses['loss_reg_cls'] = loss_out["loss"].item()


    fake_traj_rel = get_fake_traj_rel(data['traj1'], out_rel['reg'])
    pred_traj_rel = get_pred_traj_rel(data['trajs2'])
    scores_fake = discriminator(fake_traj_rel)
    discriminator_loss = g_loss_fn(scores_fake)
    loss += discriminator_loss

    loss += loss_out["loss"]
    loss += discriminator_loss

    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()

    optimizer_g.step()
    return losses






if __name__ == "__main__":
    main()



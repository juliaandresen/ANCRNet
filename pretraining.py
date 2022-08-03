import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from master.loaders import normalize, worker_init_fn, new_resample, MSLesionDataset_Pretraining
from master.losses import DiceLoss, NCCLoss, VectorFieldSmoothness
from master.networks import SpatialTransformer, NoCoRegNet


# Pretraining
if __name__ == '__main__':

    # ------------- TO DO ----------------------------
    # Add your paths here.
    path = "path/to/workingDirectory"
    rootdir = "path/to/data"
    # ------------------------------------------------

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    card = 0
    torch.cuda.set_device(card)
    device = 'cuda:' + str(card)

    # Set parameters
    batch_size = 5
    n_epochs = 200
    print_epoch = 2
    inshape = (5, 368, 512)
    n_feat = 8

    # Five-fold cross-validation
    for dataset in [0, 1, 2, 3, 4]:
        os.environ['PYTHONHASHSEED'] = str(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        best_train_loss = np.infty
        best_train_dice_loss = np.infty

        # Generate dataset for pre-training
        # Artificial new lesions are added on the fly during training
        dataset_train = MSLesionDataset_Pretraining(dataset=dataset, rootdir=rootdir, train=True, slices_used=0.5,
                                                    inshape=inshape)
        print('Generated train dataset.')
        loader_train = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  num_workers=6,
                                  pin_memory=True,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn)

        print('Initializing network, optimizer and scheduler...')
        noCoRegNet = NoCoRegNet(n_feat=n_feat, inshape=inshape, device=device).cuda()
        optimizer = torch.optim.Adam(noCoRegNet.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.8)

        # Initialize loss functions
        dice_loss_fn = DiceLoss()
        def_field_regress_loss = nn.MSELoss()

        time = '/fold' + str(dataset)
        print('Starting training at ' + time)
        os.mkdir(path + time)

        for epoch in range(n_epochs):
            print('Epoch: ' + str(epoch + 1))
            np.random.seed(epoch)
            scheduler.step()

            sum_loss = 0
            sum_dice = 0
            sum_regress = 0

            # Parameters must be trainable
            noCoRegNet.train()
            with torch.set_grad_enabled(True):
                b = 0
                # main loop to process all training samples (packed into batches)
                for batch in loader_train:

                    fixed = batch['image_followup'].cuda()
                    moving = batch['image_baseline'].cuda()
                    diff = fixed - moving

                    # Bring generated ground truth deformations into correct format
                    def_field = batch['def_field'].permute(0, 2, 1, 3, 4).cuda()
                    def_field = def_field[:, [1, 0], ...]
                    def_field[:, 0, ...] = def_field[:, 0, ...] * 184.0
                    def_field[:, 1, ...] = def_field[:, 1, ...] * 256.0

                    def_field1 = torch.zeros((fixed.shape[0], 3, inshape[0], inshape[1], inshape[2])).cuda()
                    def_field1[:, 1:, 0:1, ...] = def_field
                    def_field1[:, 1:, 1:2, ...] = def_field
                    def_field1[:, 1:, 2:3, ...] = def_field
                    def_field1[:, 1:, 3:4, ...] = def_field
                    def_field1[:, 1:, 4:5, ...] = def_field
                    def_field2 = F.interpolate(def_field1, size=(inshape[0], inshape[1]//2, inshape[2]//2), mode='trilinear')
                    def_field3 = F.interpolate(def_field1, size=(inshape[0], inshape[1]//4, inshape[2]//4), mode='trilinear')

                    label1 = batch['label'].cuda().float()
                    label2 = F.interpolate(label1, size=(inshape[0], inshape[1]//2, inshape[2]//2), mode='nearest')
                    label3 = F.interpolate(label1, size=(inshape[0], inshape[1]//4, inshape[2]//4), mode='nearest')

                    # Inputs of NCR-Net: Baseline and follow-up MRIs plus subtraction image
                    # Outputs: Velocity and diffeomorphic deformation fields, deformed baseline,
                    # segmentation of new lesions (noCoMap) and appearanace offsets (appMap);
                    # each for all three resolution levels
                    _, phi1, warped1, _, phi2, _, _, phi3, _, \
                    noCoMap1, noCoMap2, noCoMap3, appMap1, _, _ = noCoRegNet(moving=moving, fixed=fixed, diff=diff)

                    # Loss components: Regression of ground-truth deformation and Dice loss
                    r1 = def_field_regress_loss(def_field1, phi1)
                    r2 = def_field_regress_loss(def_field2, phi2)
                    r3 = def_field_regress_loss(def_field3, phi3)
                    r = 0.7 * r1 + 0.2 * r2 + 0.1 * r3

                    d1 = dice_loss_fn(noCoMap1, label1)
                    d2 = dice_loss_fn(noCoMap2, label2)
                    d3 = dice_loss_fn(noCoMap3, label3)
                    d = 0.7 * d1 + 0.2 * d2 + 0.1 * d3

                    loss = 0.05 * r + d

                    sum_regress += r.item()
                    sum_dice += d.item()
                    sum_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            train_loss = sum_loss / len(loader_train)
            reg_loss = sum_regress / len(loader_train)
            dice_loss = sum_dice / len(loader_train)

            print(datetime.datetime.now())
            print('Training Loss: {:.4f} (DF Regression: {:.4f}, Dice: {:.4f})'.format(train_loss, reg_loss, dice_loss))

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():

                    label = batch['label'].cuda().float()
                    fig, ax = plt.subplots(3, 5)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')
                    plot_slice = inshape[0]//2

                    ax[0, 0].imshow(moving[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Moving')

                    ax[1, 0].imshow(fixed[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[1, 0].title.set_text('\nFixed')

                    ax[2, 0].imshow(warped1[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray', interpolation=None,
                                    vmin=0, vmax=1)
                    ax[2, 0].title.set_text('Warped')

                    diff_img = (fixed[0, 0, plot_slice, ...] - moving[0, 0, plot_slice, ...]).cpu().detach().numpy()
                    ax[1, 1].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 1].title.set_text('Diff. before')

                    ax[0, 3].imshow(def_field1[0, 0, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[0, 3].title.set_text('GT 1')
                    ax[1, 3].imshow(def_field1[0, 1, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[1, 3].title.set_text('GT 2')
                    ax[2, 3].imshow(def_field1[0, 2, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[2, 3].title.set_text('GT 3')

                    ax[0, 4].imshow(phi1[0, 0, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[0, 4].title.set_text('Def. F. 1')
                    ax[1, 4].imshow(phi1[0, 1, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[1, 4].title.set_text('Def. F. 2')
                    ax[2, 4].imshow(phi1[0, 2, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[2, 4].title.set_text('Def. F. 3')

                    ax[1, 2].imshow(
                        (fixed[0, 0, plot_slice, ...] - warped1[0, 0, plot_slice, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 2].title.set_text('Diff. after')

                    ax[0, 1].imshow((label[0, 0, plot_slice, ...]).cpu().detach().numpy(), vmin=0, vmax=1,
                                    interpolation=None)
                    ax[0, 1].title.set_text('GT')

                    ax[0, 2].imshow(noCoMap1[0, 0, plot_slice, ...].cpu().detach().numpy(), interpolation=None,
                                    vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Segm.')

                    ax[2, 1].imshow(appMap1[0, 0, plot_slice, ...].cpu().detach().numpy(), interpolation=None, cmap='gray')
                    ax[2, 1].title.set_text('App.')

                    ax[2, 2].imshow((appMap1 * noCoMap1)[0, 0, plot_slice, ...].cpu().detach().numpy(), interpolation=None, cmap='gray')
                    ax[2, 2].title.set_text('App. * S.')

                    [axi.get_xaxis().set_visible(False) for axi in ax.ravel()]
                    [axi.get_yaxis().set_visible(False) for axi in ax.ravel()]

                    plt.savefig(path + time + "/Result_Epoch" + str(epoch + 1) + ".png")
                    plt.close()

            # Save single loss components for each epoch in CSV file
            df = pd.DataFrame(data=np.array([[train_loss, reg_loss, dice_loss]]),
                              columns=('Loss', 'MSE Deformation Field', 'Dice Loss'))
            if epoch == 0:
                df.to_csv(path + time + "/registrationMetrics.csv")
            else:
                df.to_csv(path + time + "/registrationMetrics.csv",
                          mode='a', header=False)

            # Store network parameters after every epoch
            state = {'time': str(datetime.datetime.now()),
                     'model_state': noCoRegNet.state_dict(),
                     'model_name': type(noCoRegNet).__name__,
                     'optimizer_state': optimizer.state_dict(),
                     'optimizer_name': type(optimizer).__name__,
                     'scheduler_state': scheduler.state_dict(),
                     'scheduler_name': type(scheduler).__name__,
                     'epoch': epoch,
                     'train_loss': train_loss
                     }
            torch.save(state, path + time + "/registrationNet_lastEpoch_split" + str(dataset) + ".pt")

            # Store best performing networks (we also tried to store networks performing best on validation data
            # but in the end found the networks trained until the end perform best -> faster training since
            # no calculations on val. data necessary)
            if train_loss < best_train_loss:
                torch.save(state, path + time + "/registrationNet_bestTrainLoss_split" + str(dataset) + ".pt")
                best_train_loss = train_loss
            if dice_loss < best_train_dice_loss:
                torch.save(state, path + time + "/registrationNet_bestTrainDiceLoss_split" + str(dataset) + ".pt")
                best_train_dice_loss = dice_loss

        # Plot loss once training is finished
        if not os.path.exists(path + time + "/RegistrationLosses.png"):
            print('Plotting results ...')
            df = pd.read_csv(path + time + "/registrationMetrics.csv")
            losses = df['Loss']

            plt.figure()
            plt.plot(np.arange(1, n_epochs) + 1, losses[1:])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.savefig(path + time + "/RegistrationLosses.png")
            plt.close()


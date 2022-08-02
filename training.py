import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import torch
import torch.nn.functional as F


from torch.utils.data import DataLoader
from master.loaders import normalize, worker_init_fn, new_resample, MSLesionDataset
from master.losses import DiceLoss, NCCLoss, VectorFieldSmoothness
from master.networks import SpatialTransformer, NoCoRegNet


# Main training
if __name__ == '__main__':

    # ------------- TO DO ----------------------------
    # Add your paths here.
    path = "path/to/workingDirectory"
    pretrain_path = "path/to/pretrainedNetworks"
    # ------------------------------------------------

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    card = 0
    torch.cuda.set_device(card)
    device = 'cuda:' + str(card)

    # Set parameters
    batch_size = 5
    n_epochs = 400
    print_epoch = 5
    inshape = (5, 368, 512)
    n_feat = 8

    # Initialize loss functions
    imageDist_loss_fn = NCCLoss()
    dice_loss_fn = DiceLoss()
    defField_reg_loss_fn = VectorFieldSmoothness()
    defField_reg_loss_fn2 = VectorFieldSmoothness(h=2)
    defFiel_reg_loss_fn3 = VectorFieldSmoothness(h=4)

    # Initialize transformers for different resolution levels and with different modes for deformation
    # of MRIs and segmentations
    transformer = SpatialTransformer(inshape, device)
    transformer1 = SpatialTransformer(inshape, device, mode='nearest')
    transformer2 = SpatialTransformer((inshape[0], inshape[1] // 2, inshape[2] // 2), device, mode='nearest')
    transformer3 = SpatialTransformer((inshape[0], inshape[1] // 4, inshape[2] // 4), device, mode='nearest')

    # Five-fold cross-validation
    for dataset in [0, 1, 2, 3, 4]:

        # Seeding
        os.environ['PYTHONHASHSEED'] = str(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        best_train_loss = np.infty
        best_train_dice_loss = np.infty

        # Generate train dataset
        dataset_train = MSLesionDataset(dataset=dataset, train=True, preprocessed=False,
                                        slices_used=0.00, inshape=inshape)
        print('Generated train dataset')
        loader_train = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  num_workers=6,
                                  pin_memory=True,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn)

        # Initialize network, optimizer and scheduler
        noCoRegNet = NoCoRegNet(n_feat=n_feat, inshape=inshape, device=device).cuda()
        optimizer = torch.optim.Adam(noCoRegNet.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 10, gamma=0.8)

        # Load weights generated in pre-training
        print('Loading pretrained deformable registration net...')
        noCoRegNet.load_state_dict(torch.load(os.path.join(pretrain_path,
                                                           'registrationNet_lastEpoch_split' + str(dataset) + '.pt'),
                                              map_location=device)['model_state'])
        noCoRegNet.train()

        time = '/fold' + str(dataset)
        print('Starting training at ' + time)
        os.mkdir(path + time)

        # Main training step
        for epoch in range(n_epochs):
            print('Epoch: ' + str(epoch + 1))
            np.random.seed(epoch)
            scheduler.step()

            sum_loss = 0
            sum_defField_reg = 0
            sum_imageDist = 0
            sum_dice = 0

            with torch.set_grad_enabled(True):
                for batch in loader_train:

                    followup = batch['image_followup'].cuda()
                    followup2 = F.interpolate(followup, size=(inshape[0], inshape[1]//2, inshape[2]//2), mode='trilinear')
                    followup3 = F.interpolate(followup, size=(inshape[0], inshape[1]//4, inshape[2]//4), mode='trilinear')

                    baseline = batch['image_baseline'].cuda()

                    label1 = batch['label'].cuda().float()
                    label2 = F.interpolate(label1, size=(inshape[0], inshape[1] // 2, inshape[2] // 2), mode='nearest')
                    label3 = F.interpolate(label1, size=(inshape[0], inshape[1] // 4, inshape[2] // 4), mode='nearest')

                    diff = followup - baseline

                    # Inputs of NCR-Net: Baseline and follow-up MRIs plus subtraction image
                    # Outputs: Velocity and diffeomorphic deformation fields, deformed baseline,
                    # segmentation of new lesions (noCoMap) and appearanace offsets (appMap);
                    # each for all three resolution levels
                    v1, phi1, warped1, v2, phi2, warped2, v3, phi3, warped3, \
                    noCoMap1, noCoMap2, noCoMap3, appMap1, appMap2, appMap3 = noCoRegNet(moving=baseline,
                                                                                         fixed=followup,
                                                                                         diff=diff)

                    # Loss components: Deformation field regularizer, image distance, Dice loss
                    # Each component is calculated on three resolution levels
                    reg1 = defField_reg_loss_fn(v1)
                    reg2 = defField_reg_loss_fn2(v2)
                    reg3 = defFiel_reg_loss_fn3(v3)
                    reg = 0.7 * reg1 + 0.2 * reg2 + 0.1 * reg3

                    imageDist1 = imageDist_loss_fn(warped1, followup)
                    imageDist2 = imageDist_loss_fn(warped2, followup2)
                    imageDist3 = imageDist_loss_fn(warped3, followup3)
                    imageDist = 0.7 * imageDist1 + 0.2 * imageDist2 + 0.1 * imageDist3

                    d1 = dice_loss_fn(transformer1(noCoMap1, phi1), label1)
                    d2 = dice_loss_fn(transformer2(noCoMap2, phi2), label2)
                    d3 = dice_loss_fn(transformer3(noCoMap3, phi3), label3)
                    d = 0.7 * d1 + 0.2 * d2 + 0.1 * d3

                    loss = 0.0006 * reg + 10 * imageDist + 1 * d

                    sum_defField_reg += reg.item()
                    sum_imageDist += imageDist.item()
                    sum_dice += d.item()
                    sum_loss += loss.item()

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            train_loss = sum_loss / len(loader_train)
            reg_loss = sum_defField_reg / len(loader_train)
            imageDist_loss = sum_imageDist / len(loader_train)
            dice_loss = sum_dice / len(loader_train)

            print(datetime.datetime.now())
            print('Training Loss: {:.4f} (Regularizer: {:.4f}, '
                  'Image Distance (Final): {:.4f}, Dice: {:.4f})'.format(train_loss, reg_loss, imageDist_loss, dice_loss))

            # Store figure with results every print_epoch'th epoch
            if epoch % print_epoch == 0:
                with torch.no_grad():
                    fig, ax = plt.subplots(3, 4)
                    fig.suptitle('Training, Epoch: ' + str(epoch + 1) + '\n')
                    plot_slice = inshape[0]//2

                    ax[0, 0].imshow(baseline[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray',
                                    interpolation=None, vmin=0, vmax=1)
                    ax[0, 0].title.set_text('Baseline')

                    ax[0, 1].imshow(followup[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray',
                                    interpolation=None, vmin=0, vmax=1)
                    ax[0, 1].title.set_text('\nFollow-Up')

                    ax[0, 2].imshow(warped1[0, 0, plot_slice, ...].cpu().detach().numpy(), cmap='gray',
                                    interpolation=None, vmin=0, vmax=1)
                    ax[0, 2].title.set_text('Warped Baseline')

                    diff_img = (followup[0, 0, plot_slice, ...] - baseline[0, 0, plot_slice, ...]).cpu().detach().numpy()
                    ax[1, 0].imshow(diff_img, cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 0].title.set_text('Diff. before')

                    magn = torch.sqrt(torch.square(phi1[0, 0, plot_slice, :, :]) +
                                      torch.square(phi1[0, 1, plot_slice, :, :]) +
                                      torch.square(phi1[0, 2, plot_slice, :, :])).cpu().detach().numpy()
                    ax[1, 1].imshow(magn, interpolation=None)
                    ax[1, 1].title.set_text('Def. Field')

                    ax[0, 3].imshow(phi1[0, 0, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[0, 3].title.set_text('Def. Field 1')
                    ax[1, 3].imshow(phi1[0, 1, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[1, 3].title.set_text('Def. Field 2')
                    ax[2, 3].imshow(phi1[0, 2, plot_slice, :, :].cpu().detach().numpy(), interpolation=None)
                    ax[2, 3].title.set_text('Def. Field 3')

                    ax[1, 2].imshow(
                        (followup[0, 0, plot_slice, ...] - warped1[0, 0, plot_slice, ...]).cpu().detach().numpy(),
                        cmap='gray', interpolation=None, vmin=diff_img.min(), vmax=diff_img.max())
                    ax[1, 2].title.set_text('Diff. after')

                    ax[2, 0].imshow((label1[0, 0, plot_slice, ...]).cpu().detach().numpy(), vmin=0, vmax=1,
                                    interpolation=None)
                    ax[2, 0].title.set_text('GT')

                    ax[2, 1].imshow(transformer1(noCoMap1, phi1)[0, 0, plot_slice, ...].cpu().detach().numpy(),
                                    interpolation=None, vmin=0, vmax=1)
                    ax[2, 1].title.set_text('Segmentation')

                    ax[2, 2].imshow(transformer(appMap1, phi1)[0, 0, plot_slice, ...].cpu().detach().numpy(),
                                    interpolation=None, cmap='gray')
                    ax[2, 2].title.set_text('Appearance Map')

                    plt.savefig(path + time + "/Result_Epoch" + str(epoch + 1) + ".png")
                    plt.close()

            # Save single loss components for each epoch in CSV file
            df = pd.DataFrame(data=np.array([[train_loss, reg_loss, imageDist_loss,dice_loss]]),
                              columns=('Loss', 'Regularizer Loss', 'Image Distance', 'Dice Loss'))
            if epoch == 0:
                df.to_csv(path + time + "/registrationMetrics.csv")
            else:
                df.to_csv(path + time + "/registrationMetrics.csv", mode='a', header=False)

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
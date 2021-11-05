import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils.utils import save_checkpoint
import config.config as config
# from utils.model_training import CE_Loss, Dice_loss
# from utils.utils import get_lr
# from utils.utils_metrics import f_score

def train_loop(model, gen_train, gen_val, optimizer, loss_function,  epoch, tensorboard:SummaryWriter):
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(gen_train)

    training_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0
    model.train()
    print('Start Train')
    for iteration, batch in enumerate(loop):
        # imgs, pngs, labels = batch
        imgs, labels = batch

        with torch.no_grad():
            imgs = imgs.float()
            labels = labels.float()
        imgs = imgs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        training_loss += loss.item()
        # total_f_score += _f_score.item()
        if iteration % 5 == 0:
            loop.set_postfix(
                epoch=epoch,
                training_loss=loss
            )
    tensorboard.add_scalars("loss",
                            {'training loss': training_loss},
                            epoch)

    # model.eval()
    # print('Start Validation')
    # for iteration, batch in enumerate(gen_val):
    #     imgs, labels = batch
    #     with torch.no_grad():
    #         imgs = imgs.float()
    #         labels = labels.float()
    #         if config.CUDA:
    #             imgs = imgs.to(config.DEVICE)
    #             labels = labels.to(config.DEVICE)
    #
    #         outputs = model(imgs)
    #         loss = loss_function(outputs, labels)
    #         # -------------------------------#
    #         #   计算f_score
    #         # -------------------------------#
    #         # _f_score = f_score(outputs, labels)
    #
    #         val_loss += loss.item()
    #         # val_f_score += _f_score.item()
    #
    # tensorboard.add_scalars("loss",
    #                         {'valing loss': val_loss},
    #                         epoch)

# def do_train(model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
#              cuda, dice_loss, num_classes):
#
#     total_loss = 0
#     total_f_score = 0
#
#     val_loss = 0
#     val_f_score = 0
#
#     model.train()
#     print('Start Train')
#     with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen):
#             if iteration >= epoch_step:
#                 break
#             imgs, pngs, labels = batch
#             with torch.no_grad():
#                 imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
#                 pngs = torch.from_numpy(pngs).long()
#                 labels = torch.from_numpy(labels).type(torch.FloatTensor)
#                 if cuda:
#                     imgs = imgs.cuda()
#                     pngs = pngs.cuda()
#                     labels = labels.cuda()
#
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = CE_Loss(outputs, pngs, num_classes=num_classes)
#             if dice_loss:
#                 main_dice = Dice_loss(outputs, labels)
#                 loss = loss + main_dice
#             with torch.no_grad():
#                 # -------------------------------#
#                 #   计算f_score
#                 # -------------------------------#
#                 _f_score = f_score(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_f_score += _f_score.item()
#
#             pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
#                                 'f_score': total_f_score / (iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#     print('Finish Train')
#
#     model.eval()
#     print('Start Validation')
#     with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen_val):
#             if iteration >= epoch_step_val:
#                 break
#             imgs, pngs, labels = batch
#             with torch.no_grad():
#                 imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
#                 pngs = torch.from_numpy(pngs).long()
#                 labels = torch.from_numpy(labels).type(torch.FloatTensor)
#                 if cuda:
#                     imgs = imgs.cuda()
#                     pngs = pngs.cuda()
#                     labels = labels.cuda()
#
#                 outputs = model(imgs)
#                 loss = CE_Loss(outputs, pngs, num_classes=num_classes)
#                 if dice_loss:
#                     main_dice = Dice_loss(outputs, labels)
#                     loss = loss + main_dice
#                 # -------------------------------#
#                 #   计算f_score
#                 # -------------------------------#
#                 _f_score = f_score(outputs, labels)
#
#                 val_loss += loss.item()
#                 val_f_score += _f_score.item()
#
#             pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
#                                 'f_score': val_f_score / (iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#     loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
#     print('Finish Validation')
#     print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
#     print('Total Loss: %.3f || Val Loss: %.3f ' % (
#     total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
#     torch.save(model.state_dict(), 'weights/ep%03d-loss%.3f-val_loss%.3f.pth' % (
#         (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))

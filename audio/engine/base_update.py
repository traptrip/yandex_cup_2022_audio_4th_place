import os

import torch
import numpy as np
from tqdm import tqdm

import utils as lib


def _batch_optimization(
    config,
    net,
    batch,
    criterion,
    optimizer,
    loss_optimizer,
    scaler,
    epoch,
    memory
):
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        di = net(batch["features"].cuda())
        labels = batch["label"].cuda()
        scores = torch.mm(di, di.t())
        label_matrix = lib.create_label_matrix(labels)

        if memory:
            memory_embeddings, memory_labels = memory(di.detach(), labels, batch["path"])
            if epoch >= config.memory.activate_after:
                memory_scores = torch.mm(di, memory_embeddings.t())
                memory_label_matrix = lib.create_label_matrix(labels, memory_labels)

        logs = {}
        losses = []
        for crit, weight in criterion:
            if hasattr(crit, 'takes_embeddings'):
                loss = crit(di, labels.view(-1))
                if memory:
                    if epoch >= config.memory.activate_after:
                        mem_loss = crit(di, labels.view(-1), memory_embeddings, memory_labels.view(-1))

            else:
                loss = crit(scores, label_matrix)
                if memory:
                    if epoch >= config.memory.activate_after:
                        mem_loss = crit(memory_scores, memory_label_matrix)

            loss = loss.mean()
            if weight == 'adaptative':
                losses.append(loss)
            else:
                losses.append(weight * loss)

            logs[crit.__class__.__name__] = loss.item()
            if memory:
                if epoch >= config.memory.activate_after:
                    mem_loss = mem_loss.mean()
                    if weight == 'adaptative':
                        losses.append(config.memory.weight * mem_loss)
                    else:
                        losses.append(weight * config.memory.weight * mem_loss)
                    logs[f"memory_{crit.__class__.__name__}"] = mem_loss.item()

    if weight == 'adaptative':
        grads = []
        for i, lss in enumerate(losses):
            g = torch.autograd.grad(lss, net.fc.parameters(), retain_graph=True)
            grads.append(torch.norm(g[0]).item())
        mean_grad = np.mean(grads)
        weights = [mean_grad / g for g in grads]
        losses = [w * lss for w, lss in zip(weights, losses)]
        logs.update({
            f"weight_{crit.__class__.__name__}": w for (crit, _), w in zip(criterion, weights)
        })
        logs.update({
            f"grad_{crit.__class__.__name__}": w for (crit, _), w in zip(criterion, grads)
        })

    total_loss = sum(losses)
    if scaler is None:
        total_loss.backward()
    else:
        scaler.scale(total_loss).backward()

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_update(
    config,
    net,
    loader,
    criterion,
    optimizer,
    loss_optimizer,
    scheduler,
    scaler,
    epoch,
    memory=None,
):
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()
    if len(loss_optimizer) > 0:
        _ = [crit.zero_grad() for crit, w in criterion]

    iterator = tqdm(loader, disable=os.getenv('TQDM_DISABLE'))
    for i, batch in enumerate(iterator):
        logs = _batch_optimization(
            config,
            net,
            batch,
            criterion,
            optimizer,
            loss_optimizer,
            scaler,
            epoch,
            memory,
        )

        if config.experience.log_grad:
            grad_norm = lib.get_gradient_norm(net)
            logs["grad_norm"] = grad_norm.item()

        for key, opt in optimizer.items():
            if epoch < config.experience.warm_up and key != config.experience.warm_up_key:
                lib.LOGGER.info(f"Warming up @epoch {epoch}")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)

        for key, opt in loss_optimizer.items():
            if epoch < config.experience.warm_up and key != config.experience.warm_up_key:
                lib.LOGGER.info(f"Warming up @epoch {epoch}")
                continue
            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]

        for sch in scheduler["on_step"]:
            sch.step()

        if scaler is not None:
            scaler.update()

        meter.update(logs)
        if not os.getenv('TQDM_DISABLE'):
            iterator.set_postfix(meter.avg)
        else:
            if (i + 1) % 50 == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loader)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')

    for crit, _ in criterion:
        if hasattr(crit, 'step'):
            crit.step()

    return meter.avg

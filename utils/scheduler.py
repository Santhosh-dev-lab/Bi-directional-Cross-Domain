import torch.optim as optim


def load_scheduler(model_name, model, opts):
    optimizer, scheduler = None, None

    if model_name == 'm3ddcnn':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = None

    elif model_name == 'cnn3d':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'rssan':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.0)
        scheduler = None

    elif model_name == 'ablstm':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)

    elif model_name == 'dffn':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'speformer':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)

    elif model_name in ('ssftt', 'BiDA'):
        lr = getattr(opts, 'lr', 1e-2)
        weight_decay = getattr(opts, 'weight_decay', 1e-4)
        momentum = getattr(opts, 'momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        sched_type = getattr(opts, 'scheduler', 'none')
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epoch, eta_min=1e-5)
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(opts.epoch * 0.5), int(opts.epoch * 0.75)], gamma=0.1)
        else:
            scheduler = None

    elif model_name == 'GAHT':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        scheduler = None

    return optimizer, scheduler

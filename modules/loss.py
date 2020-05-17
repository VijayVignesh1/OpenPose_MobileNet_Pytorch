def l2_loss(input, target, batch_size):
    loss = (input - target)
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()

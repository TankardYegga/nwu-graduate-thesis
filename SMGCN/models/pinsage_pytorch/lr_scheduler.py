from torch.optim.lr_scheduler import _LRScheduler

class Polynominal(_LRScheduler):

    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(Polynominal, self).__init__(optimizer, last_epoch)

    # 定义学习率衰减的策略
    def polynominal_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    # 定义学习率的更新策略
    def get_lr(self):
        # 满足一定条件学习率原来是什么还是什么
        if (
                (self.last_epoch == 0)
            or (self.last_epoch % self.step_size !=0)
            or (self.last_epoch > self.iter_max)
           ):
            return [group['lr'] for group in self.optimizer.param_groups]
        print('base lrs is:', self.base_lrs)
        # 不然则对每个已知的lr应用新策略
        return [self.polynominal_decay(lr) for lr in self.base_lrs]


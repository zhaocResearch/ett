from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./logs")
for i in range(100):
    writer.add_scalar(tag='test2*i', scalar_value=2*i, global_step=i)
writer.close()
import torch
import torch.nn as nn
from torch import optim

'''
Self Created Imports
'''
from layout_gn import Encoder, Generator, Discriminator


def train(generator, discriminator, dataloader, num_epochs, device, lr=0.0002, beta1=0.5, beta2=0.999):
    
  g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
  d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
  criterion = nn.BCELoss()
  fixed_noise = torch.randn(64, 128, 1, 1, device=device)

  for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
      batch_size = real_images.size(0)
      real_images = real_images.to(device)

      # 训练判别器
      d_optimizer.zero_grad()
      # 真实样本的标签为1
      real_labels = torch.ones(batch_size, device=device)
      # 生成器生成的样本的标签为0
      fake_labels = torch.zeros(batch_size, device=device)

      # 计算真实样本在判别器上的损失
      real_outputs = discriminator(real_images)
      d_loss_real = criterion(real_outputs, real_labels)

      # 计算生成器生成的样本在判别器上的损失
      noise = torch.randn(batch_size, 128, 1, 1, device=device)
      fake_images = generator(noise)
      fake_outputs = discriminator(fake_images.detach())
      d_loss_fake = criterion(fake_outputs, fake_labels)

      # 将真实样本和生成的样本的损失相加作为判别器的总损失，并更新判别器的参数
      d_loss = d_loss_real + d_loss_fake
      d_loss.backward()
      d_optimizer.step()

      # 训练生成器
      g_optimizer.zero_grad()
      # 生成器生成的样本的标签为1
      fake_labels.fill_(1)
      # 计算生成器生成的样本在判别器上的损失，并更新生成器的参数
      fake_outputs = discriminator(fake_images)
      g_loss = criterion(fake_outputs, fake_labels)
      g_loss.backward()
      g_optimizer.step()

      # 打印损失信息
      if i % 100 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
          % (epoch, num_epochs, i, len(dataloader),
          d_loss.item(), g_loss.item()))

    # 保存生成器的输出结果
    fake_images = generator(fixed_noise)
    save_image(fake_images.detach(), 'fake_samples_epoch_%03d.png' % epoch, normalize=True)

    # 保存模型参数
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

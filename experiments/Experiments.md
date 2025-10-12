# Experiments

List of all experiments and data

## Enhancer

### DENSE 64x1 - 48x1 - 32x1 - 16x1

epoch=98-v16.ckpt

wandb:           g_l1_loss 0.00892
wandb:              g_loss 0.0116
wandb:          g_mse_loss 0.00024
wandb:       g_msssim_loss 0.03029
wandb:         g_ssim_loss 0.04907
wandb:          val_g_loss 0.15216
wandb:            val_psnr 36.15864
wandb:        val_ref_psnr 36.14266
wandb:        val_ref_ssim 0.94966
wandb:            val_ssim 0.95105

View run giddy-bee-1190 at: https://wandb.ai/doman/vvc-enhancer/runs/1jjrds5u
Duration
48m 21s

### DENSE 128x1 - 64x1 - 48x1 - 16x1

epoch=98-v9.ckpt

wandb:           g_l1_loss 0.00778
wandb:              g_loss 0.00972
wandb:          g_mse_loss 0.00035
wandb:       g_msssim_loss 0.02494
wandb:         g_ssim_loss 0.0398
wandb:          val_g_loss 0.21625
wandb:            val_psnr 36.22152
wandb:        val_ref_psnr 36.25641
wandb:        val_ref_ssim 0.95003
wandb:            val_ssim 0.95189

View run lively-capybara-1178 at: https://wandb.ai/doman/vvc-enhancer/runs/pdovhy4
Duration
1h 15m 49s

### DENSE 16x8 - 16x4 - 16x3 - 16x1

epoch=98-v10.ckpt

wandb:           g_l1_loss 0.00429
wandb:              g_loss 0.00473
wandb:          g_mse_loss 7e-05
wandb:       g_msssim_loss 0.01188
wandb:         g_ssim_loss 0.01794
wandb:          val_g_loss 0.17276
wandb:            val_psnr 35.99114
wandb:        val_ref_psnr 35.98959
wandb:        val_ref_ssim 0.94889
wandb:            val_ssim 0.94935

View run comic-planet-1179 at: https://wandb.ai/doman/vvc-enhancer/runs/4eyxo761
Duriation
4h 45m 51s

### RAW 128x1 - 64x1 - 32x1 - 16x1

epoch=98-v13.ckpt

wandb:           g_l1_loss 0.02923
wandb:              g_loss 0.03279
wandb:          g_mse_loss 0.00176
wandb:       g_msssim_loss 0.06062
wandb:         g_ssim_loss 0.14336
wandb:          val_g_loss 0.27615
wandb:            val_psnr 26.49849
wandb:        val_ref_psnr 36.33117
wandb:        val_ref_ssim 0.94976
wandb:            val_ssim 0.9215

View run dulcet-haze-1186 at: https://wandb.ai/doman/vvc-enhancer/runs/zeq4jtyr
Duration
38m 7s

### RESNET 64x1 - 48x1 - 32x1 - 16x1

epoch=98-v14.ckpt

wandb:           g_l1_loss 0.00879
wandb:              g_loss 0.01277
wandb:          g_mse_loss 0.00028
wandb:       g_msssim_loss 0.03644
wandb:         g_ssim_loss 0.05503
wandb:          val_g_loss 0.3489
wandb:            val_psnr 35.53786
wandb:        val_ref_psnr 35.52991
wandb:        val_ref_ssim 0.94508
wandb:            val_ssim 0.94541

View run fast-bee-1187 at: https://wandb.ai/doman/vvc-enhancer/runs/etuvcsv7
Duration
42m 57s

## DENSE without mask

epoch=98-v22.ckpt

wandb:           g_l1_loss 0.05399
wandb:              g_loss 0.03454
wandb:          g_mse_loss 0.00572
wandb:       g_msssim_loss 0.0448
wandb:         g_ssim_loss 0.06176
wandb:       val_g_l1_loss 0.02089
wandb:          val_g_loss 0.01897
wandb:      val_g_mse_loss 0.00081
wandb:   val_g_msssim_loss 0.0371
wandb:     val_g_ssim_loss 0.06585
wandb:            val_psnr 30.23544
wandb:        val_ref_psnr 35.97268
wandb:        val_ref_ssim 0.94849
wandb:            val_ssim 0.92855

View run sage-leaf-1238 at: https://wandb.ai/doman/vvc-enhancer/runs/08nx1n2h
Duration
48m 27s

## Discriminator

### classic one 4 -> 1

epoch=98-v19.ckpt

wandb:         d_fake_loss 0.12268
wandb:              d_loss 0.08121
wandb:         d_real_loss 0.03973
wandb:      val_d_fakeloss 0.1586
wandb:          val_d_loss 0.15162
wandb:     val_d_real_loss 0.14465

View run jumping-durian-1217 at: https://wandb.ai/doman/vvc-enhancer/runs/bstwildi
Duration
54m 29s

### DenseNet

epoch=98-v18.ckpt

wandb:         d_fake_loss 0.00205
wandb:              d_loss 0.00169
wandb:         d_real_loss 0.00133
wandb:      val_d_fakeloss 0.21481
wandb:          val_d_loss 0.24708
wandb:     val_d_real_loss 0.27935

View run chocolate-water-1196 at: https://wandb.ai/doman/vvc-enhancer/runs/c3mtxkca
Duration
1h 7m 43s

# RESNET

epoch=98-v21.ckpt

wandb:         d_fake_loss 0.00361
wandb:              d_loss 0.01605
wandb:         d_real_loss 0.02849
wandb:     val_d_fake_loss 0.19794
wandb:          val_d_loss 0.26163
wandb:     val_d_real_loss 0.32532

View run solar-glade-1235 at: https://wandb.ai/doman/vvc-enhancer/runs/1yj3ehha
Duration
23m 8s

## Discriminator

### normal one

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 128, 66, 66]           6,144
       BatchNorm2d-2          [-1, 128, 66, 66]             256
         LeakyReLU-3          [-1, 128, 66, 66]               0
            Conv2d-4          [-1, 256, 33, 33]         524,288
       BatchNorm2d-5          [-1, 256, 33, 33]             512
         LeakyReLU-6          [-1, 256, 33, 33]               0
            Conv2d-7          [-1, 512, 16, 16]       2,097,152
       BatchNorm2d-8          [-1, 512, 16, 16]           1,024
         LeakyReLU-9          [-1, 512, 16, 16]               0
           Conv2d-10           [-1, 1024, 8, 8]       8,388,608
      BatchNorm2d-11           [-1, 1024, 8, 8]           2,048
        LeakyReLU-12           [-1, 1024, 8, 8]               0
           Conv2d-13           [-1, 2048, 4, 4]      33,554,432
      BatchNorm2d-14           [-1, 2048, 4, 4]           4,096
        LeakyReLU-15           [-1, 2048, 4, 4]               0
           Conv2d-16              [-1, 1, 1, 1]          32,768
          Flatten-17                    [-1, 1]               0
          Sigmoid-18                    [-1, 1]               0
================================================================
Total params: 44,611,328
Trainable params: 44,611,328
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.20
Forward/backward pass size (MB): 24.39
Params size (MB): 170.18
Estimated Total Size (MB): 194.77
----------------------------------------------------------------


wandb: Run summary:
wandb:         d_fake_loss 0.0
wandb:              d_loss 0.00022
wandb:         d_real_loss 0.00045
wandb:               epoch 98
wandb:             lr-Adam 0.0
wandb:              lr-SGD 1e-05
wandb: trainer/global_step 49999
wandb:     val_d_fake_loss 0.23371
wandb:          val_d_loss 0.21284
wandb:     val_d_real_loss 0.19197
wandb:
wandb: 🚀 View run revived-puddle-1263 at: https://wandb.ai/doman/vvc-enhancer/runs/lytz5qy5


ANOTHER ONE

wandb: Run summary:
wandb:               d_acc 0.9375
wandb:          d_fake_acc 0.875
wandb:         d_fake_loss 0.1213
wandb:              d_loss 0.07475
wandb:          d_real_acc 1.0
wandb:         d_real_loss 0.0282
wandb:               epoch 198
wandb:             lr-Adam 0.0
wandb:              lr-SGD 0.0
wandb: trainer/global_step 99999
wandb:           val_d_acc 0.60312
wandb:      val_d_fake_acc 0.70417
wandb:     val_d_fake_loss 0.2107
wandb:          val_d_loss 0.23908
wandb:      val_d_real_acc 0.50208
wandb:     val_d_real_loss 0.26746
wandb:
wandb: 🚀 View run twilight-moon-1272 at: https://wandb.ai/doman/vvc-enhancer/runs/1dxqsxjc
wandb: Synced 5 W&B file(s), 12623 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230731_211125-1dxqsxjc/logs
python3 -m enhancer -e 200 -m "discriminator"  7897.76s user 502.28s system 130% cpu 1:47:14.29 total
[doman@doman-pc ~/Projects/Studies/mgr]$ lls checkpoints | head             [git][mainU]
Permissions Size User  Group Date Modified Name
.rw-r--r--  358M doman doman 31 Jul 22:58   epoch=198-v8.ckpt

DENSE

wandb: Run summary:
wandb:               d_acc 1.0
wandb:          d_fake_acc 1.0
wandb:         d_fake_loss 0.01149
wandb:              d_loss 0.00673
wandb:          d_real_acc 1.0
wandb:         d_real_loss 0.00196
wandb:               epoch 198
wandb:             lr-Adam 0.0
wandb:              lr-SGD 0.0
wandb: trainer/global_step 99999
wandb:           val_d_acc 0.46042
wandb:      val_d_fake_acc 0.02083
wandb:     val_d_fake_loss 0.27828
wandb:          val_d_loss 0.24517
wandb:      val_d_real_acc 0.9
wandb:     val_d_real_loss 0.21205
wandb:
wandb: 🚀 View run smart-lion-1273 at: https://wandb.ai/doman/vvc-enhancer/runs/0tbyabon
wandb: Synced 5 W&B file(s), 12624 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230801_194829-0tbyabon/logs
python3 -m enhancer -e 200 -m "discriminator"  8974.41s user 928.31s system 123% cpu 2:13:21.37 total
.rw-r--r--   57M doman doman  1 Aug 22:01   epoch=198-v9.ckpt

ResNet


wandb: Run summary:
wandb:               d_acc 0.875
wandb:          d_fake_acc 0.75
wandb:         d_fake_loss 0.00329
wandb:              d_loss 0.00324
wandb:          d_real_acc 1.0
wandb:         d_real_loss 0.00319
wandb:               epoch 198
wandb:             lr-Adam 0.0
wandb:              lr-SGD 0.0
wandb: trainer/global_step 99999
wandb:           val_d_acc 0.49479
wandb:      val_d_fake_acc 0.88125
wandb:     val_d_fake_loss 0.21864
wandb:          val_d_loss 0.24997
wandb:      val_d_real_acc 0.10833
wandb:     val_d_real_loss 0.28131
wandb:
wandb: 🚀 View run fragrant-armadillo-1274 at: https://wandb.ai/doman/vvc-enhancer/runs/w8ds610d
wandb: Synced 5 W&B file(s), 12624 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230801_230823-w8ds610d/logs
python3 -m enhancer -e 200 -m "discriminator"  4245.82s user 469.93s system 162% cpu 48:16.03 total
[doman@doman-pc ~/Projects/Studies/mgr]$ lls checkpoints | head             [git][mainU]
Permissions Size User  Group Date Modified Name
.rw-r--r--   90M doman doman  1 Aug 23:56   epoch=198-v10.ckpt


# GENERATOR

## Dense

wandb: Run summary:
wandb:               epoch 998
wandb:           g_l1_loss 0.00669
wandb:              g_loss 0.00816
wandb:          g_mse_loss 9e-05
wandb:       g_msssim_loss 0.01829
wandb:         g_ssim_loss 0.03622
wandb:             lr-Adam 0.0
wandb:              lr-SGD 0.0
wandb: trainer/global_step 499999
wandb:       val_g_l1_loss 0.00868
wandb:          val_g_loss 0.0114
wandb:      val_g_mse_loss 0.00025
wandb:   val_g_msssim_loss 0.02964
wandb:     val_g_ssim_loss 0.04863
wandb:            val_psnr 36.09806
wandb:        val_ref_psnr 35.27636
wandb:        val_ref_ssim 0.94917
wandb:            val_ssim 0.96257
wandb:
wandb: 🚀 View run driven-star-1283 at: https://wandb.ai/doman/vvc-enhancer/runs/7k4tg4pf
wandb: Synced 5 W&B file(s), 94535 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230803_085908-7k4tg4pf/logs
python3 -m enhancer -e 1000 -m "enhancer"  37306.51s user 2454.89s system 137% cpu 8:02:49.38 total
[doman@doman-pc ~/Projects/Studies/mgr]$ lls checkpoints | head -n 2                                                                                                                                                                                                                                                                                                 [git][mainU]
Permissions Size User  Group Date Modified Name
.rw-r--r--   47M doman doman  3 Aug 17:01   epoch=998-v10.ckpt

## ResNet

wandb: Run summary:
wandb:               epoch 998
wandb:           g_l1_loss 0.01141
wandb:              g_loss 0.01864
wandb:          g_mse_loss 0.00044
wandb:       g_msssim_loss 0.04908
wandb:         g_ssim_loss 0.08997
wandb:             lr-Adam 0.0
wandb:              lr-SGD 0.0
wandb: trainer/global_step 499999
wandb:       val_g_l1_loss 0.00873
wandb:          val_g_loss 0.01163
wandb:      val_g_mse_loss 0.00028
wandb:   val_g_msssim_loss 0.03012
wandb:     val_g_ssim_loss 0.05012
wandb:            val_psnr 35.59898
wandb:        val_ref_psnr 35.57854
wandb:        val_ref_ssim 0.9494
wandb:            val_ssim 0.95149
wandb:
wandb: 🚀 View run noble-cloud-1282 at: https://wandb.ai/doman/vvc-enhancer/runs/fulh4cjs
wandb: Synced 5 W&B file(s), 94535 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230802_225606-fulh4cjs/logs
python3 -m enhancer -e 1000 -m "enhancer"  33986.29s user 2398.79s system 141% cpu 7:09:24.28 total
[doman@doman-pc ~/Projects/Studies/mgr]$ lls checkpoints | head -n 2                                                                                                 [git][mainU]
Permissions Size User  Group Date Modified Name
.rw-r--r--   48M doman doman  3 Aug 06:05   epoch=998-v9.ckpt

## Conv DIS

wandb: Run history:
wandb:               d_acc ▃▃▅▇▄▄▃▆█▆▇▅▇▄█▅▇▇▇▆▇▇▅▇▆▅▅▄▁▆▇▇▇▄▅▆▄▅▇█
wandb:          d_fake_acc ▁▁▄█▅▇▂▇▇▇█▅▇▇▇██▇▅▇█▇▅██▄█▇▄▇█▇█▅█▇▂▇█▅
wandb:         d_fake_loss ▆▅▅▃▃▂█▂▃▂▁▃▁▃▂▂▂▂▃▃▂▁▄▂▃▄▂▂▃▂▁▂▂▄▂▃▄▂▂▅
wandb:              d_loss ▅▅▅▃▅▄▇▆▁▃▁▄▂█▁▅▃▃▂▃▃▂▇▃▄▃▄▄█▄▂▂▁▆▅▆▂▅▄▁
wandb:          d_real_acc ▆▆▆▅▃▂▅▅▇▅▅▅▆▂▇▂▅▆▇▅▅▆▅▅▃▆▂▂▁▅▅▆▅▃▂▅▆▃▅█
wandb:         d_real_loss ▄▄▄▄▅▅▅▇▃▄▃▄▄█▃▆▄▄▃▄▄▄▇▄▅▃▅▅█▅▄▃▃▆▆▆▃▆▅▁
wandb:               epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             lr-Adam ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:              lr-SGD ████████▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: trainer/global_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:           val_d_acc ▁▁▆████▇█████▇▇████▇█▇▇███
wandb:      val_d_fake_acc ▁▃▇█▆▇█▇▇▇▇▇▆▆▇▇▇▇▇▇▇▇▇▇▇▆
wandb:     val_d_fake_loss ██▃▂▃▃▁▂▂▃▃▂▃▃▃▂▂▂▃▃▂▃▃▃▂▃
wandb:          val_d_loss ██▅▃▂▂▁▂▂▂▁▂▁▁▂▁▁▂▁▂▁▂▁▁▁▂
wandb:      val_d_real_acc ▂▁▅▆█▇▇▆▇██▇██▇█▇▇▇▇▇▇▇▇▇█
wandb:     val_d_real_loss ██▇▄▂▂▃▃▂▁▁▂▁▁▂▁▁▂▂▂▁▂▁▁▂▁
wandb:
wandb: Run summary:
wandb:               d_acc 0.8125
wandb:          d_fake_acc 1.0
wandb:         d_fake_loss 0.15543
wandb:              d_loss 0.18592
wandb:          d_real_acc 0.625
wandb:         d_real_loss 0.21642
wandb:               epoch 98
wandb:             lr-Adam 0.0001
wandb:              lr-SGD 0.0
wandb: trainer/global_step 49999
wandb:           val_d_acc 0.80104
wandb:      val_d_fake_acc 0.875
wandb:     val_d_fake_loss 0.18159
wandb:          val_d_loss 0.1623
wandb:      val_d_real_acc 0.72708
wandb:     val_d_real_loss 0.14301


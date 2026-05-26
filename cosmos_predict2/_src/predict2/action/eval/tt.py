import lpips

# 按你实际会用的 backbone 准备
for net in ["alex", "vgg", "squeeze"]:
    print("loading", net)
    loss_fn = lpips.LPIPS(net=net)
print("done")
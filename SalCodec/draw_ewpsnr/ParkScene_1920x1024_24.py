import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'ParkScene_1920x1024_24'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.0253, 0.1074, 0.2461, 0.3471], [32.684116, 37.217065, 39.689513, 41.234797]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.016159413, 0.041452407, 0.099415518, 0.276466477], [32.61627907, 35.51162791, 38.47674419, 41.54651163]
cqy, = plt.plot(bpp, psnr, "c-o", color=config.color.CQY, linewidth=LineWidth, label=config.label.CQY)
tb.add_column("CQY", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.022482661, 0.056206653, 0.13524726, 0.362884206], [32.26744186, 35.1627907, 38.26744186, 41.3372093]
ldp, = plt.plot(bpp, psnr, "c-o", color=config.color.LDP, linewidth=LineWidth, label=config.label.LDP)
tb.add_column("LDP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.022482661, 0.056206653, 0.13313951, 0.338645087], [31.77906977, 34.56976744, 37.39534884, 40.29069767]
aqp, = plt.plot(bpp, psnr, "c-o", color=config.color.AQP, linewidth=LineWidth, label=config.label.AQP)
tb.add_column("AQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.023536536, 0.058314403, 0.141570508, 0.381853952], [32.26744186, 35.19767442, 38.26744186, 41.44186047]
mqp, = plt.plot(bpp, psnr, "c-o", color=config.color.MQP, linewidth=LineWidth, label=config.label.MQP)
tb.add_column("MQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.050581012, 0.077341403, 0.118080677, 0.180265816], [33.0855, 34.4709, 35.9285, 37.1672]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.035663883, 0.056553308, 0.091489133, 0.14680735], [33.2568, 34.7269, 36.2204, 37.468]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

print(tb)
savepathpsnr = prefix + '/ParkScene_1920x1024_24'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, cqy, Lu_TPAMI2021, DCVC, ldp, aqp, mqp], loc=4)
# plt.legend(handles=[h264, h265, DVC, DVCp, eccv, iccv, EA, RY, Liu, LU, rafc, FVC, ELF, DCVC], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('EWPSNR (dB)')
plt.title('ParkScene_1920x1024_24')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()



# savepath = prefix + '/' + 'UVG' + '.png'
# img1 = cv2.imread(savepathpsnr + '.png')

# image = np.concatenate((img1, img2), axis=1)
# cv2.imwrite(savepath, image)
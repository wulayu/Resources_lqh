import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'FourPeople_1280x720_60'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.0033051, 0.018682, 0.049708, 0.12577], [34.268, 37.8584, 40.706, 42.5606]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.007017341, 0.013615736, 0.033725131, 0.1103922], [34.18604651, 37.25581395, 40.08139535, 43.08139535]
cqy, = plt.plot(bpp, psnr, "c-o", color=config.color.CQY, linewidth=LineWidth, label=config.label.CQY)
tb.add_column("CQY", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.005446295, 0.01204469, 0.02964041, 0.114162712], [33.87209302, 36.94186047, 39.73255814, 42.62790698]
ldp, = plt.plot(bpp, psnr, "c-o", color=config.color.LDP, linewidth=LineWidth, label=config.label.LDP)
tb.add_column("LDP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.006703132, 0.012358899, 0.029011992, 0.095938573], [33.20930233, 36.20930233, 38.93023256, 41.61627907]
aqp, = plt.plot(bpp, psnr, "c-o", color=config.color.AQP, linewidth=LineWidth, label=config.label.AQP)
tb.add_column("AQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.006388922, 0.012358899, 0.02995462, 0.112905875], [33.62790698, 36.55813953, 39.45348837, 42.41860465]
mqp, = plt.plot(bpp, psnr, "c-o", color=config.color.MQP, linewidth=LineWidth, label=config.label.MQP)
tb.add_column("MQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.0315522, 0.042684, 0.0578012, 0.082617], [36.3792, 37.7859, 39.1344, 40.0633]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.018498372, 0.027411178, 0.037974498, 0.058584506], [36.2374, 37.8643, 39.3261, 40.0387]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

print(tb)
savepathpsnr = prefix + '/' + prefix
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, cqy, Lu_TPAMI2021, DCVC, ldp, aqp, mqp], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('EWPSNR (dB)')
plt.title(prefix)
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()



# savepath = prefix + '/' + 'UVG' + '.png'
# img1 = cv2.imread(savepathpsnr + '.png')

# image = np.concatenate((img1, img2), axis=1)
# cv2.imwrite(savepath, image)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'BasketballDrive_1920x1024_50'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.0327, 0.0759, 0.1386, 0.3195], [36.764858, 38.194691, 39.51936, 42.781616]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.031513568, 0.057297396, 0.136081316, 0.373865512], [35.4801444, 37.13718412, 39.57400722, 43.44043321]
cqy, = plt.plot(bpp, psnr, "c-o", color=config.color.CQY, linewidth=LineWidth, label=config.label.CQY)
tb.add_column("CQY", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.030081133, 0.058729831, 0.126054272, 0.465541346], [34.14801444, 36.22743682, 38.17689531, 41.13357401]
ldp, = plt.plot(bpp, psnr, "c-o", color=config.color.LDP, linewidth=LineWidth, label=config.label.LDP)
tb.add_column("LDP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.031513568, 0.064459571, 0.15470297, 0.538595526], [33.72563177, 35.80505415, 37.85198556, 40.51624549]
aqp, = plt.plot(bpp, psnr, "c-o", color=config.color.AQP, linewidth=LineWidth, label=config.label.AQP)
tb.add_column("AQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.031513568, 0.061594701, 0.138946186, 0.528568482], [34.08303249, 36.1299639, 38.17689531, 41.26353791]
mqp, = plt.plot(bpp, psnr, "c-o", color=config.color.MQP, linewidth=LineWidth, label=config.label.MQP)
tb.add_column("MQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.056984688, 0.078334564, 0.146620719, 0.444079473], [35.0906, 36.2771, 38.1891, 40.8218]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])


bpp, psnr = [0.027333574, 0.060023226, 0.158421065, 0.389914229], [35.6138, 36.6075, 39.2236, 41.5011]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])



print(tb)
savepathpsnr = prefix + '/BasketballDrive_1920x1024_50'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, cqy, Lu_TPAMI2021, DCVC, ldp, aqp, mqp], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('EWPSNR (dB)')
plt.title('BasketballDrive_1920x1024_50')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()


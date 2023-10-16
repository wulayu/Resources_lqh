import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'KristenAndSara_1280x720_60'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.010361, 0.023273, 0.044167, 0.070075], [37.2936, 40.0341, 41.5006, 42.5144]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.00406901, 0.008138021, 0.020345052, 0.078125], [35.24378109, 37.80845771, 39.93432836, 43.960199]
cqy, = plt.plot(bpp, psnr, "c-o", color=config.color.CQY, linewidth=LineWidth, label=config.label.CQY)
tb.add_column("CQY", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.004340278, 0.00922309, 0.025770399, 0.095486111], [33.97014925, 37.07462687, 40.13930348, 42.88557214]
ldp, = plt.plot(bpp, psnr, "c-o", color=config.color.LDP, linewidth=LineWidth, label=config.label.LDP)
tb.add_column("LDP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.004611545, 0.01030816, 0.027398003, 0.098741319], [33.09452736, 36.11940299, 39.06467662, 41.73134328]
aqp, = plt.plot(bpp, psnr, "c-o", color=config.color.AQP, linewidth=LineWidth, label=config.label.AQP)
tb.add_column("AQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.004340278, 0.009494358, 0.026041667, 0.095214844], [34.08955224, 37.23383085, 40.13930348, 42.92537313]
mqp, = plt.plot(bpp, psnr, "c-o", color=config.color.MQP, linewidth=LineWidth, label=config.label.MQP)
tb.add_column("MQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.026861453, 0.035920755, 0.049555677, 0.071378843], [37.234, 38.8351, 39.9937, 41.58]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.014823681, 0.021620316, 0.030908451, 0.048830151], [37.0901, 38.7107, 39.968, 40.6115]
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


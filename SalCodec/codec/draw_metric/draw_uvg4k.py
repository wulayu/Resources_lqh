import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'UVG4Kresults'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])
bpp_, psnr_ = [0.01625616, 0.0311656, 0.052445, 0.08021682], [37.6515, 39.61566, 40.841235,41.616516]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.017846159409317705, 0.02883415639667806, 0.05872785419881068, 0.12140161527872638], [37.571536085340706, 38.904855235417686, 39.92576607598199, 41.29943616655138]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.010392705, 0.016096245, 0.046231225, 0.171424649], [37.04285, 38.31555, 39.6612, 41.26628571]
H265, = plt.plot(bpp, psnr, "c-o", color="darkorange", linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.01048968, 0.019700039, 0.03246367, 0.082730517], [37.925225, 39.443125, 40.64085, 42.1635]
H266, = plt.plot(bpp, psnr, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H.266", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

savepathpsnr = prefix + '/UVG_psnr'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, DCVC, H266, H265], loc=4)
# plt.legend(handles=[h264, h265, DVC, DVCp, eccv, iccv, EA, RY, Liu, LU, rafc, FVC, ELF, DCVC], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('PSNR (dB)')
plt.title('4K Sequences Test')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()
print(tb)

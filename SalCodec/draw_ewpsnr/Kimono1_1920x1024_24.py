import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'Kimono1_1920x1024_24'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.0197, 0.0729, 0.125, 0.1663], [39.619888, 42.172485, 43.513979, 43.933369]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.017997755, 0.032959743, 0.062883721, 0.139645228], [37.91058394, 40.20985401, 41.90693431, 43.63138686]
cqy, = plt.plot(bpp, psnr, "c-o", color=config.color.CQY, linewidth=LineWidth, label=config.label.CQY)
tb.add_column("CQY", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.03100818, 0.055077466, 0.103216038, 0.216406736], [37.74635036, 40.04562044, 41.79744526, 43.08394161]
ldp, = plt.plot(bpp, psnr, "c-o", color=config.color.LDP, linewidth=LineWidth, label=config.label.LDP)
tb.add_column("LDP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.029707137, 0.056378508, 0.107119166, 0.234621331], [37.58211679, 39.90875912, 41.66058394, 42.94708029]
aqp, = plt.plot(bpp, psnr, "c-o", color=config.color.AQP, linewidth=LineWidth, label=config.label.AQP)
tb.add_column("AQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.03100818, 0.058980593, 0.111022293, 0.232019246], [37.74635036, 40.10036496, 41.82481752, 43.05656934]
mqp, = plt.plot(bpp, psnr, "c-o", color=config.color.MQP, linewidth=LineWidth, label=config.label.MQP)
tb.add_column("MQP", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.056654865, 0.078810897, 0.109108202, 0.151503673], [38.768, 40.3024, 41.7834, 42.8989]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.037968855, 0.055055224, 0.080680436, 0.117158914], [38.5847, 40.8437, 42.0529, 43.0174]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])


print(tb)
savepathpsnr = prefix + '/Kimono1_1920x1024_24'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, cqy, Lu_TPAMI2021, DCVC, ldp, aqp, mqp], loc=4)
# plt.legend(handles=[h264, h265, DVC, DVCp, eccv, iccv, EA, RY, Liu, LU, rafc, FVC, ELF, DCVC], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('EWPSNR (dB)')
plt.title('Kimono1_1920x1024_24')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()



# savepath = prefix + '/' + 'UVG' + '.png'
# img1 = cv2.imread(savepathpsnr + '.png')

# image = np.concatenate((img1, img2), axis=1)
# cv2.imwrite(savepath, image)
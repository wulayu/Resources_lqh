import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'MCLresults'
LineWidth = config.svg.LineWidth
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [0.042904656, 0.072172949, 0.152439024, 0.211862528], [34.15492958, 35.25352113, 37.1971831, 37.77464789]
Ours, = plt.plot(bpp_, psnr_, "c-o", color="red", linewidth=LineWidth, label='Ours')

bpp, psnr = [0.04616307, 0.075899281, 0.137290168, 0.213549161], [34.39846743, 35.82375479, 37.54022989, 38.59770115]
V1, = plt.plot(bpp, psnr, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.06707989, 0.086088154, 0.126859504, 0.204545455], [34.8967033, 36.03956044, 37.24395604, 38.36043956]
Liu_TCSVT2022, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.04338843, 0.076721763, 0.160192837, 0.200137741], [34.10549451, 35.65274725, 37.19120879, 37.74505495]
H265, = plt.plot(bpp, psnr, "c-o", color="gold", linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.065977961, 0.093801653, 0.152203857, 0.26046832], [34.68571429, 36.31208791, 37.55164835, 38.78241758]
Agustsson_CVPR2020, = plt.plot(bpp, psnr, "c-o", color="saddlebrown", linewidth=LineWidth, label=config.label.Agustsson_CVPR2020)
tb.add_column("Agustsson_CVPR2020", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.049449036, 0.075344353, 0.116942149, 0.20399449], [33.64835165, 35.38021978, 36.6021978, 38.03516484]
Yang_JSTSP2021, = plt.plot(bpp, psnr, "c-o", color="thistle", linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

print(tb)
savepathpsnr = prefix + '/MCL_psnr' + '.png'
savepathpsnrsvg = prefix + '/MCL_psnr' + '.svg'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, V1, Liu_TCSVT2022, Agustsson_CVPR2020, Yang_JSTSP2021, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR (dB)')
plt.title('MCL-JCV dataset')
plt.savefig(savepathpsnr)
plt.savefig(savepathpsnrsvg, format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-MSSSIM", "BD-Rate"])
print("----MSSSIM----")

bpp_, msssim_  = [0.092325056, 0.118848758, 0.192776524, 0.262189616], [0.971969697, 0.974939394, 0.979, 0.981]
Ours, = plt.plot(bpp_, msssim_, "c-o", color="red", linewidth=LineWidth, label='Ours')

bpp, msssim  = [0.059878419, 0.098176292, 0.169908815, 0.265349544], [0.96824359, 0.976897436, 0.983307692, 0.987730769]
V1, = plt.plot(bpp, msssim , "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim  = [0.058688525, 0.086229508, 0.186557377, 0.312459016], [0.965947712, 0.974313725, 0.983006536, 0.987287582]
Liu_TCSVT2022, = plt.plot(bpp, msssim , "c-o", color="royalblue", linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim  = [0.075737705, 0.152459016, 0.249836066, 0.33442623], [0.96627451, 0.974542484, 0.978627451, 0.982058824]
H265, = plt.plot(bpp, msssim , "c-o", color="gold", linewidth=LineWidth, label='H.265')
tb.add_column("H.265", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim  = [0.06, 0.083934426, 0.123278689, 0.228852459], [0.965686275, 0.973496732, 0.977320261, 0.984836601]
Agustsson_CVPR2020, = plt.plot(bpp, msssim , "c-o", color="saddlebrown", linewidth=LineWidth, label=config.label.Agustsson_CVPR2020)
tb.add_column("Agustsson_CVPR2020", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim  = [0.069836066, 0.119344262, 0.184918033, 0.329508197], [0.967941176, 0.975784314, 0.981797386, 0.986339869]
Yang_JSTSP2021, = plt.plot(bpp, msssim , "c-o", color="thistle", linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])
print(tb)

savepathmsssim = prefix + '/' + 'MCL_msssim' + '.png'
savepathmsssimsvg = prefix + '/' + 'MCL_msssim' + '.svg'
plt.legend(handles=[Ours, V1, Liu_TCSVT2022, Agustsson_CVPR2020, Yang_JSTSP2021, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('MCL-JCV dataset')
plt.savefig(savepathmsssim)
plt.savefig(savepathmsssimsvg, format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()


savepath = prefix + '/' + 'MCL' + '.png'
img1 = cv2.imread(savepathpsnr)
img2 = cv2.imread(savepathmsssim)

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)
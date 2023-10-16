import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'UVGresults'
plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth
print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [ 0.0431,0.0647,0.1469,0.2079 ], [ 35.173,36.4454,38.024,38.687 ]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, psnr = [0.037377049, 0.061967213, 0.09429, 0.132786885], [35.04564315, 36.30290456, 37.40467, 38.05809129]
V1, = plt.plot(bpp, psnr, "c-o", color=config.color.V1, linewidth=LineWidth, label='V1')
tb.add_column("Old", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.050645161, 0.063870968, 0.096774194, 0.157419355], [35.00684932, 35.75342466, 36.83561644, 37.94520548]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.053870968, 0.079032258, 0.122580645, 0.206451613], [35.0890411, 36.17808219, 37.16438356, 38.23287671]
Liu_TCSVT2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.048709677, 0.072580645, 0.107096774, 0.182903226], [34.96575342, 36.17123288, 37.33561644, 38.45890411]
Liu_TCSVT2022, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.060749186, 0.077361564, 0.108631922, 0.185179153], [34.55909944, 35.19399625, 36.68667917, 37.69136961]
DVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.051628664, 0.077361564, 0.12752443, 0.230130293], [34.53939962, 35.85928705, 37.19230769, 38.45309568]
Yang_JSTSP2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Yang_JSTSP2021, linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.023020134, 0.038724832, 0.071342282, 0.142214765], [34.57718121, 35.70469799, 36.84563758, 38.01342282]
H266, = plt.plot(bpp, psnr, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H.266", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.02709702142196456, 0.04098199165086189, 0.06370908663667266, 0.10385984934969908], [34.011881197066536, 35.20922313871838, 36.15551458086286, 37.444169357844764]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.014820847, 0.074104235, 0.165309446, 0.296905537], [34.46060038, 35.90525328, 37.30393996, 38.17729831]
H265, = plt.plot(bpp, psnr, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

# bpp, psnr = [0.036970684, 0.083876221, 0.165635179, 0.292996743], [34.07317073, 35.05159475, 36.20731707, 37.17917448]
# H264, = plt.plot(bpp, psnr, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])
print(tb)
savepathpsnr = prefix + '/UVG_psnr'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, Yang_JSTSP2021, DVC, DCVC, H266, H265], loc=4)
# plt.legend(handles=[h264, h265, DVC, DVCp, eccv, iccv, EA, RY, Liu, LU, rafc, FVC, ELF, DCVC], loc=4)
plt.grid()
plt.xlabel('BPP')
plt.ylabel('PSNR (dB)')
plt.title('UVG dataset')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathpsnr + '.png')
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-MSSSIM", "BD-Rate"])
print("----MSSSIM----")
bpp_, msssim_ = [ 0.0431,0.0647,0.1469,0.2079 ], [ 0.961,0.969,0.977,0.979 ]
Ours, = plt.plot(bpp_, msssim_, "c-o", color=config.color.Ours, linewidth=LineWidth, label='Ours')

bpp, msssim = [0.04505618, 0.106348315, 0.211460674, 0.29511236], [0.960733788, 0.975418089, 0.98331058, 0.986327645]
V1, = plt.plot(bpp, msssim, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.074514039, 0.100431965, 0.161771058, 0.277969762], [0.959701031, 0.966752577, 0.974175258, 0.98042268]
Lu_TPAMI2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.083585313, 0.126781857, 0.208855292, 0.30475162], [0.967, 0.973, 0.979989691, 0.983020619]
Liu_TCSVT2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.057235421, 0.093088553, 0.196328294, 0.377321814], [0.96143299, 0.971268041, 0.980608247, 0.987350515]
Liu_TCSVT2022, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.061555076, 0.109071274, 0.186393089, 0.239956803], [0.949556701, 0.965515464, 0.971391753, 0.975597938]
DVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.076673866, 0.113822894, 0.212742981, 0.383801296], [0.964030928, 0.96928866, 0.977886598, 0.983082474]
Yang_JSTSP2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Yang_JSTSP2021, linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.065471167, 0.117510549, 0.213853727, 0.333755274], [0.965238532, 0.97459633, 0.982027523, 0.987201835]
DCVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.022247423, 0.078680412, 0.12814433, 0.231752577], [0.950366972, 0.9744091743, 0.98, 0.985798165]
H266, = plt.plot(bpp, msssim, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H.266", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.015766739, 0.074514039, 0.166522678, 0.394168467], [0.950051546, 0.960381443, 0.970340206, 0.979804124]
H265, = plt.plot(bpp, msssim, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

# bpp, msssim = [0.038660907, 0.08488121, 0.188552916, 0.396760259], [0.947391753, 0.956298969, 0.968051546, 0.976402062]
# H264, = plt.plot(bpp, msssim, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])
print(tb)

savepathmsssim = prefix + '/' + 'UVG_msssim'# + '.svg'
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, Yang_JSTSP2021, DVC, DCVC, H266, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('UVG dataset')
plt.savefig(savepathmsssim + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.savefig(savepathmsssim + '.png')
plt.clf()


savepath = prefix + '/' + 'UVG' + '.png'
img1 = cv2.imread(savepathpsnr + '.png')
img2 = cv2.imread(savepathmsssim + '.png')

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)
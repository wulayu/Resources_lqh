import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from BD import BD_PSNR, BD_RATE
import prettytable as pt
from config import config

prefix = 'JCT720presult'

plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth

print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])

bpp_, psnr_ = [ 0.0332,0.0552,0.1198,0.1345 ], [ 36.606,38.214,39.877,40.178 ]
Ours, = plt.plot(bpp_, psnr_, "c-o", color=config.color.Ours, linewidth=LineWidth, label=config.label.Ours)

bpp, psnr = [0.030391061, 0.045072046, 0.063687151, 0.084022346], [36.34951456, 37.91304348, 39.34951456, 40.23786408]
V1, = plt.plot(bpp, psnr, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.030991597, 0.043361345, 0.061781513, 0.107092437], [36.31073446, 37.67608286, 38.78719397, 39.95480226]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.03394958, 0.047260504, 0.067966387, 0.112336134], [36.82862524, 38.00564972, 38.97551789, 39.96421846]
Liu_TCSVT2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.029243697, 0.043630252, 0.060436975, 0.094722689], [36.33081285, 37.78638941, 39.02457467, 40.15879017]
Liu_TCSVT2022, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.036638655, 0.049680672, 0.070789916, 0.121747899], [36.01883239, 37.51600753, 38.68361582, 39.80414313]
DVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.036835443, 0.050102672, 0.08049226, 0.10684951], [37.360746, 38.4705151, 40.198490231, 40.829069272]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.0324375, 0.0555, 0.0844375, 0.1121875], [37.26174497, 39.32080537, 40.5557047, 41.14228188]
H266, = plt.plot(bpp, psnr, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H.266", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.034487395, 0.062184874, 0.095798319, 0.133042017], [36.58601134, 38.16446125, 38.89224953, 39.64839319]
H265, = plt.plot(bpp, psnr, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

# bpp, psnr = [0.036773109, 0.061781513, 0.095932773, 0.125915966], [35.3100189, 37.06805293, 37.91871456, 38.69376181]
# H264, = plt.plot(bpp, psnr, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])
print(tb)

savepathpsnr = prefix + '/JCT-VC-720p_psnr'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, DVC, DCVC, H266, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR (dB)')
plt.title('JCT-VC(720p) dataset')
plt.savefig(savepathpsnr + '.png')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-MSSSIM", "BD-Rate"])
print("----MSSSIM----")
bpp_, msssim_ = [ 0.0328,0.0696,0.0925,0.1421 ], [ 0.982,0.987,0.989,0.991 ]
Ours, = plt.plot(bpp_, msssim_, "c-o", color=config.color.Ours, linewidth=LineWidth, label=config.label.Ours)

bpp, msssim = [0.035078125, 0.051953125, 0.099609375, 0.150390625], [0.983666667, 0.987142857, 0.99016, 0.9919261905]
V1, = plt.plot(bpp, msssim, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.035059932, 0.048972603, 0.074229452, 0.126455479], [0.980732044, 0.984958564, 0.987914365, 0.990400552]
Lu_TPAMI2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.022217466, 0.041481164, 0.080436644, 0.162842466], [0.982030387, 0.984654696, 0.987969613, 0.991837017]
Liu_TCSVT2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.029279601, 0.042760342, 0.082988588, 0.181847361], [0.982794118, 0.986130515, 0.98921875, 0.992444853]
Liu_TCSVT2022, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.03677226, 0.049828767, 0.070804795, 0.121532534], [0.976892265, 0.983024862, 0.986450276, 0.988660221]
DVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.025469169, 0.049162198, 0.106400804, 0.190790885], [0.982885191, 0.986677205, 0.98997005, 0.991497837]
DCVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.021633238, 0.034527221, 0.060315186, 0.13252149], [0.98291954, 0.986733333, 0.988602299, 0.991671264]
H266, = plt.plot(bpp, msssim, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H.266", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.034415121, 0.061804565, 0.133273894, 0.197681883], [0.9803125, 0.984503676, 0.987536765, 0.988501838]
H265, = plt.plot(bpp, msssim, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

# bpp, msssim = [0.036558219, 0.062457192, 0.125813356, 0.199657534], [0.976339779, 0.982279006, 0.98609116, 0.987361878]
# H264, = plt.plot(bpp, msssim, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])
print(tb)
savepathmsssim = prefix + '/' + 'JCT-VC-720p_msssim'# + '.svg'
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, DVC, DCVC, H266, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('JCT-VC(720p) dataset')
plt.savefig(savepathmsssim + '.png')
plt.savefig(savepathmsssim + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()


savepath = prefix + '/' + 'JCT-VC-720p' + '.png'
img1 = cv2.imread(savepathpsnr + '.png')
img2 = cv2.imread(savepathmsssim + '.png')

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)
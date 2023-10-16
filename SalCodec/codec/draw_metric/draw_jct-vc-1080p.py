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

prefix = 'JCT1080presult'

plt.figure(dpi=config.svg.dpi)
matplotlib.rc('font', **config.font)
LineWidth = config.svg.LineWidth


print("----PSNR----")
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-PSNR", "BD-Rate"])
bpp_, psnr_ = [0.069631902, 0.08803681, 0.119325153, 0.161656442], [32.7672956, 33.3836478, 34.1509434, 34.81761006]
Ours, = plt.plot(bpp_, psnr_, "c-o", color="red", linewidth=LineWidth, label='Ours')

bpp, psnr = [0.067884131, 0.118261965, 0.168639798, 0.25743073], [32.5795053, 33.97879859, 34.72791519, 35.54770318]
V1, = plt.plot(bpp, psnr, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.063898305, 0.091355932, 0.153728814, 0.281525424], [32.1451049, 33.26398601, 34.3041958, 35.32692308]
Lu_TPAMI2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.078813559, 0.120169492, 0.201186441, 0.347627119], [32.87062937, 33.80594406, 34.86363636, 35.82517483]
Liu_TCSVT2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.075423729, 0.116779661, 0.189661017, 0.330677966], [32.61713287, 33.7972028, 34.92482517, 35.93006993]
Liu_TCSVT2022, = plt.plot(bpp, psnr, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.076779661, 0.115423729, 0.165932203, 0.293389831], [31.9527972, 33.08916084, 34.14685315, 35.12587413]
DVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.049051348, 0.079943517, 0.128415918, 0.186824134], [31.61514683, 32.8438949, 33.96445131, 34.8763524]
DCVC, = plt.plot(bpp, psnr, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.047336683, 0.065477387, 0.105628141, 0.168944724], [31.36700337, 32.92592593, 33.98989899, 35.06734007]
H266, = plt.plot(bpp, psnr, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H266", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.069322034, 0.108983051, 0.187627119, 0.346610169], [31.82167832, 33.30769231, 34.62762238, 35.7465035]
Yang_JSTSP2021, = plt.plot(bpp, psnr, "c-o", color=config.color.Yang_JSTSP2021, linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

bpp, psnr = [0.063220339, 0.115084746, 0.25779661, 0.347627119], [31.97027972, 33.36888112, 34.81118881, 35.22202797]
H265, = plt.plot(bpp, psnr, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])

# bpp, psnr = [0.064915254, 0.116779661, 0.248644068, 0.347966102], [31.13986014, 32.61713287, 34.02447552, 34.53146853]
# H264, = plt.plot(bpp, psnr, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, psnr, bpp_, psnr_), BD_RATE(bpp, psnr, bpp_, psnr_)])
print(tb)
savepathpsnr = prefix + '/JCT-VC-1080p_psnr'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, Yang_JSTSP2021, DVC, DCVC, H266, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR (dB)')
plt.title('JCT-VC(1080p) dataset')
plt.savefig(savepathpsnr + '.png')
plt.savefig(savepathpsnr + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
tb = pt.PrettyTable()
tb.set_style(pt.PLAIN_COLUMNS)
tb.add_column("", ["BD-MSSSIM", "BD-Rate"])
print("----MSSSIM----")
bpp_, msssim_ = [0.070623145, 0.087537092, 0.11958457, 0.161424332], [0.962970297, 0.965874587, 0.970891089, 0.974983498]
Ours, = plt.plot(bpp_, msssim_, "c-o", color="red", linewidth=LineWidth, label='Ours')

bpp, msssim = [0.072807018, 0.135964912, 0.253508772, 0.38245614], [0.961224913, 0.972512111, 0.9793391, 0.983629758]
V1, = plt.plot(bpp, msssim, "c-o", color=config.color.V1, linewidth=LineWidth, label=config.label.V1)
tb.add_column("V1", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.085349127, 0.11415212, 0.178740648, 0.321009975], [0.956474954, 0.965231911, 0.973024119, 0.979925788]
Lu_TPAMI2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Lu_TPAMI2021, linewidth=LineWidth, label=config.label.Lu_TPAMI2021)
tb.add_column("Lu_TPAMI2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.073566085, 0.123753117, 0.214962594, 0.310972569], [0.959220779, 0.96812616, 0.976215213, 0.980296846]
Liu_TCSVT2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2021, linewidth=LineWidth, label=config.label.Liu_TCSVT2021)
tb.add_column("Liu_TCSVT2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.063528678, 0.101932668, 0.224127182, 0.45105985], [0.957068646, 0.966938776, 0.977847866, 0.985862709]
Liu_TCSVT2022, = plt.plot(bpp, msssim, "c-o", color=config.color.Liu_TCSVT2022, linewidth=LineWidth, label=config.label.Liu_TCSVT2022)
tb.add_column("Liu_TCSVT2022", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.076184539, 0.115897756, 0.166957606, 0.293952618], [0.946679035, 0.959220779, 0.968274583, 0.973914657]
DVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DVC, linewidth=LineWidth, label=config.label.DVC)
tb.add_column("DVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.060883905, 0.099472296, 0.182585752, 0.292744063], [0.95483304, 0.963885764, 0.973954306, 0.980210896]
DCVC, = plt.plot(bpp, msssim, "c-o", color=config.color.DCVC, linewidth=LineWidth, label=config.label.DCVC)
tb.add_column("DCVC", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.020480549, 0.062356979, 0.12208238, 0.26006865], [0.937668712, 0.962331288, 0.97306135, 0.980871166]
H266, = plt.plot(bpp, msssim, "c-o", color=config.color.H266, linewidth=LineWidth, label=config.label.H266)
tb.add_column("H266", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.085785536, 0.132044888, 0.231982544, 0.423566085], [0.959591837, 0.966790353, 0.976289425, 0.982745826]
Yang_JSTSP2021, = plt.plot(bpp, msssim, "c-o", color=config.color.Yang_JSTSP2021, linewidth=LineWidth, label=config.label.Yang_JSTSP2021)
tb.add_column("Yang_JSTSP2021", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

bpp, msssim = [0.060910224, 0.114588529, 0.257294264, 0.456733167], [0.948979592, 0.961150278, 0.970426716, 0.975027829]
H265, = plt.plot(bpp, msssim, "c-o", color=config.color.H265, linewidth=LineWidth, label=config.label.H265)
tb.add_column("H.265", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])

# bpp, msssim = [0.066583541, 0.116770574, 0.248129676, 0.457169576], [0.939554731, 0.955732839, 0.966790353, 0.97257885]
# H264, = plt.plot(bpp, msssim, "c-o", color=config.color.H264, linewidth=LineWidth, label=config.label.H264)
# tb.add_column("H.264", [BD_PSNR(bpp, msssim, bpp_, msssim_), BD_RATE(bpp, msssim, bpp_, msssim_)])
print(tb)
savepathmsssim = prefix + '/' + 'JCT-VC-1080p_msssim'# + '.svg'
plt.legend(handles=[Ours, V1, Lu_TPAMI2021, Liu_TCSVT2021, Liu_TCSVT2022, Yang_JSTSP2021, DVC, DCVC, H266, H265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('JCT-VC(1080p) dataset')
plt.savefig(savepathmsssim + '.png')
plt.savefig(savepathmsssim + '.svg', format='svg', dpi=config.svg.dpi, bbox_inches=config.svg.bbox_inches)
plt.clf()


savepath = prefix + '/' + 'JCT-VC-1080p' + '.png'
img1 = cv2.imread(savepathpsnr + '.png')
img2 = cv2.imread(savepathmsssim + '.png')

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)
from easydict import EasyDict

font = EasyDict(family='Times New Roman',
            weight='normal', size=12)
label = EasyDict(Ours='Ours',
            V1='V1', 
            Lu_TPAMI2021='Lu_TPAMI2021', 
            Liu_TCSVT2021='Liu_TCSVT2021',
            Liu_TCSVT2022='Liu_TCSVT2022',
            Agustsson_CVPR2020='Agustsson_CVPR2020',
            Yang_JSTSP2021='Yang_JSTSP2021',
            DVC='DVC',
            DCVC="DCVC",
            H264='H.264',
            H265='H.265',
            H266="H.266")
color = EasyDict(
    Ours="red",
    V1="hotpink",
    DCVC="green",
    Lu_TPAMI2021="dimgrey",
    Liu_TCSVT2021="blueviolet",
    Liu_TCSVT2022="gold",
    DVC="darkorange",
    Yang_JSTSP2021="cyan",
    H266="blue",
    H265="royalblue",
    # H264="hotpink",
    Agustsson_CVPR2020="saddlebrown",
)
svg = EasyDict(dpi=200,
           bbox_inches='tight',
           LineWidth=2)

config = EasyDict(font=font,
            label=label,
            svg=svg,
            color=color)
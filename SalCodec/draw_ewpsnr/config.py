from easydict import EasyDict

font = EasyDict(family='Times New Roman',
            weight='normal', size=12)
label = EasyDict(Ours='Ours',
            CQY='CQY',
            LDP='HM(LDP)', 
            AQP='AQP', 
            MQP='MQP',
            Lu_TPAMI2021='Lu_TPAMI2021',
            DCVC='DCVC'
            )
color = EasyDict(
    Ours="red",
    CQY='blue',
    LDP="hotpink",
    AQP="green",
    MQP="dimgrey",
    Lu_TPAMI2021='blueviolet',
    DCVC='cyan'
)
svg = EasyDict(dpi=200,
           bbox_inches='tight',
           LineWidth=2)

config = EasyDict(font=font,
            label=label,
            svg=svg,
            color=color)
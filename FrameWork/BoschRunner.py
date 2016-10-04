import TrainModel
import Orchestrator

from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics


import numpy as np
import pandas as pd
#from GrupoData import GrupoData
import pandas as pd
import numpy as np
import re
import random
import pickle
import os
import math
import gc

config = set()
cols =  ['L3_S38_F3952','L0_S0_F0','L1_S24_F1512','L1_S24_F1514','L3_S38_F3956','L0_S12_F330','L0_S0_F2','L1_S24_F1516','L3_S38_F3960','L1_S24_F1723','L1_S24_F1518','L1_S24_F1846','L1_S24_F1520','L0_S0_F4','L0_S12_F332','L3_S32_F3850','L0_S0_F16','L2_S26_F3036','L0_S0_F6','L2_S26_F3047','L3_S36_F3920','L3_S33_F3859','L0_S0_F20','L1_S25_F1855','L3_S30_F3759','L1_S24_F1118','L1_S24_F1539','L0_S0_F18','L3_S30_F3554','L3_S29_F3327','L2_S26_F3062','L1_S24_F1565','L3_S30_F3829','L3_S30_F3744','L0_S0_F10','L1_S24_F1575','L3_S30_F3804','L0_S1_F24','L1_S24_F1567','L3_S29_F3351','L3_S30_F3519','L0_S2_F48','L3_S29_F3333','L3_S29_F3318','L2_S27_F3199','L2_S26_F3051','L1_S24_F679','L2_S26_F3106','L3_S30_F3754','L2_S26_F3117','L2_S26_F3073','L3_S30_F3544','L0_S4_F104','L3_S29_F3373','L3_S29_F3348','L3_S35_F3889','L3_S29_F3315','L3_S30_F3609','L0_S15_F397','L3_S30_F3494','L3_S30_F3774','L0_S11_F286','L1_S24_F1578','L3_S29_F3342','L3_S30_F3809','L3_S30_F3534','L3_S29_F3407','L3_S33_F3865','L3_S29_F3473','L1_S24_F1690','L1_S24_F867','L3_S29_F3339','L1_S24_F1632','L1_S24_F1581','L3_S30_F3749','L0_S4_F109','L3_S29_F3336','L3_S30_F3604','L2_S26_F3121','L0_S10_F229','L3_S30_F3704','L0_S0_F8','L1_S24_F1685','L3_S33_F3857','L0_S2_F64','L0_S3_F80','L3_S30_F3709','L3_S29_F3345','L3_S30_F3769','L1_S24_F1126','L0_S3_F84','L1_S25_F1858','L2_S26_F3069','L0_S3_F72','L0_S21_F477','L3_S29_F3476','L0_S12_F334','L3_S29_F3330','L3_S33_F3855','L1_S24_F1695','L1_S24_F1798','L0_S7_F138','L0_S13_F356','L3_S30_F3629','L3_S30_F3514','L1_S24_F1544','L0_S2_F36','L2_S26_F3040','L3_S29_F3324','L3_S30_F3799','L3_S29_F3321','L3_S30_F3574','L1_S24_F1829','L1_S24_F1778','L0_S11_F318','L0_S12_F336','L1_S25_F2016','L0_S14_F370','L1_S25_F2231','L2_S27_F3129','L1_S24_F1713','L0_S1_F28','L1_S25_F2767','L2_S26_F3113','L0_S6_F132','L0_S0_F22','L3_S30_F3624','L0_S5_F114','L1_S24_F1672','L3_S29_F3379','L1_S24_F1014','L1_S24_F1758','L0_S6_F122','L0_S0_F12','L0_S12_F346','L1_S24_F1451','L1_S24_F1844','L2_S28_F3259','L1_S24_F1842','L0_S10_F249','L0_S9_F190','L1_S24_F1700','L1_S24_F1850','L1_S24_F1571','L0_S0_F14','L1_S24_F1637','L3_S29_F3354','L0_S2_F60','L1_S24_F1793','L1_S24_F1662','L3_S29_F3461','L3_S49_F4236','L3_S30_F3639','L2_S27_F3210','L1_S24_F1824','L0_S14_F362','L0_S21_F487','L0_S15_F415','L0_S3_F96','L0_S10_F219','L0_S9_F180','L1_S24_F1371','L3_S34_F3876','L1_S24_F1820','L2_S27_F3214','L0_S12_F350','L1_S24_F1652','L1_S24_F1498','L0_S10_F274','L1_S25_F2158','L1_S24_F1667','L0_S12_F348','L0_S5_F116','L1_S24_F1803','L0_S11_F282','L3_S29_F3357','L1_S24_F691','L3_S29_F3479','L1_S25_F2437','L3_S34_F3882','L0_S3_F100','L1_S24_F1763','L1_S25_F2594','L3_S30_F3579','L1_S24_F1122','L1_S25_F2287','L0_S9_F160','L3_S30_F3509','L0_S18_F439','L2_S27_F3162','L0_S12_F338','L1_S24_F1808','L1_S25_F1919','L0_S10_F244','L3_S30_F3669','L3_S30_F3524','L1_S25_F2233','L1_S24_F1812','L0_S23_F623','L0_S10_F224','L0_S7_F142','L1_S24_F1810','L1_S24_F1818','L3_S30_F3559','L3_S29_F3458','L0_S9_F155','L0_S10_F259','L1_S24_F1594','L3_S30_F3664','L3_S40_F3982','L1_S24_F1822','L1_S24_F687','L0_S16_F421','L0_S18_F449','L3_S30_F3794','L3_S29_F3449','L1_S24_F1773','L0_S2_F44','L3_S29_F3488','L1_S24_F1728','L0_S16_F426','L3_S36_F3918','L3_S29_F3382','L2_S28_F3307','L3_S35_F3894','L2_S27_F3155','L1_S24_F1569','L3_S30_F3784','L0_S23_F671','L1_S25_F2632','L0_S22_F591','L0_S11_F302','L1_S24_F1166','L1_S24_F1604','L0_S10_F264','L1_S24_F1361','L1_S24_F1599','L3_S40_F3984','L0_S23_F663','L1_S24_F1768','L1_S24_F1130','L0_S8_F146','L1_S24_F1753','L3_S47_F4138','L3_S29_F3370','L0_S17_F433','L0_S14_F386','L0_S9_F210','L0_S11_F306','L0_S13_F354','L3_S29_F3376','L3_S30_F3689','L3_S30_F3634','L3_S36_F3938','L0_S9_F170','L1_S24_F1738','L1_S25_F2066','L1_S24_F1743','L1_S25_F2051','L1_S24_F1622','L2_S27_F3133','L1_S24_F683','L1_S25_F2207','L1_S24_F1788','L3_S36_F3926','L3_S36_F3924','L2_S27_F3144','L1_S25_F1881','L1_S24_F1298','L3_S30_F3499','L1_S25_F1885','L1_S24_F1814','L0_S23_F627','L3_S29_F3427','L3_S29_F3360','L1_S24_F1783','L0_S22_F586','L1_S24_F1609','L0_S9_F195','L3_S29_F3433','L3_S35_F3903','L1_S25_F2837','L1_S24_F1356','L1_S25_F2161','L1_S24_F1816','L1_S24_F1184','L0_S10_F234','L0_S10_F254','L1_S24_F1275','L1_S25_F1900','L1_S24_F1134','L3_S30_F3584','L3_S29_F3401','L1_S24_F1056','L2_S27_F3166','L3_S30_F3764','L3_S29_F3412','L0_S19_F455','L3_S30_F3564','L3_S30_F3569','L3_S30_F3819','L1_S24_F1148','L1_S24_F1748','L0_S11_F290','L0_S9_F165','L3_S41_F4026','L0_S15_F418','L1_S25_F1869','L0_S15_F403','L0_S22_F561','L1_S24_F1831','L1_S24_F1573','L3_S30_F3539','L0_S23_F631','L3_S29_F3430','L1_S25_F2046','L0_S3_F92','L1_S25_F2443','L3_S40_F3980','L3_S44_F4118','L0_S14_F374','L0_S19_F459','L3_S30_F3684','L0_S12_F352','L3_S49_F4211','L3_S41_F4014','L3_S29_F3491','L1_S25_F2787','L1_S25_F2637','L1_S25_F2783','L1_S24_F1733','L3_S29_F3395','L1_S24_F1836','L1_S25_F1992','L3_S30_F3674','L1_S25_F2307','L2_S27_F3140','L0_S22_F606','L3_S49_F4221','L1_S24_F1396','L0_S23_F643','L1_S25_F2423','L0_S22_F551','L3_S30_F3504','L1_S24_F872','L1_S25_F2613','L3_S30_F3644','L0_S21_F472','L1_S25_F2498','L1_S25_F1892','L3_S44_F4112','L0_S22_F556','L1_S24_F1161','L0_S22_F546','L2_S27_F3218','L1_S24_F988','L1_S24_F1406','L1_S25_F2056','L0_S11_F294','L3_S35_F3896','L0_S21_F502','L3_S30_F3649','L3_S48_F4196','L1_S24_F1441','L1_S25_F2237','L1_S24_F1838','L3_S43_F4065','L3_S47_F4158','L0_S2_F32','L0_S21_F512','L1_S25_F2239','L2_S27_F3206','L2_S28_F3226','L1_S25_F1978','L1_S24_F1506','L3_S30_F3589','L3_S41_F4000','L0_S22_F596','L0_S17_F431','L1_S24_F1010','L1_S25_F2086','L1_S24_F1145','L3_S39_F3976','L0_S12_F342','L1_S25_F2500','L1_S25_F1997','L1_S25_F2965','L1_S25_F2365','L0_S9_F185','L1_S25_F1968','L3_S29_F3404','L3_S29_F3442','L0_S21_F517','L1_S25_F2071','L3_S41_F4020','L1_S24_F1180','L3_S29_F3455','L1_S25_F2247']
cols = ['L3_S32_F3850','L1_S24_F1672','L1_S24_F1581','L1_S24_F1571','L1_S24_F1846','L3_S33_F3855','L3_S33_F3865','L3_S29_F3407','L3_S33_F3857','L3_S29_F3412','L3_S38_F3956','L3_S40_F3986','L3_S38_F3952','L1_S24_F1844','L3_S29_F3461','L3_S29_F3467','L3_S38_F3960','L3_S29_F3327','L3_S29_F3351','L3_S29_F3370','L3_S30_F3604','L3_S34_F3876','L3_S29_F3336','L3_S30_F3809','L3_S30_F3524','L1_S24_F1565','L3_S33_F3859','L0_S1_F28','L1_S24_F1569','L3_S29_F3373','L1_S24_F1723','L3_S30_F3504','L3_S30_F3554','L3_S29_F3330','L3_S29_F3458','L3_S29_F3436','L3_S43_F4080','L1_S24_F1632','L3_S33_F3863','L0_S0_F16','L1_S24_F1838','L3_S34_F3882','L1_S24_F1842','L3_S41_F4016','L2_S26_F3069','L2_S26_F3117','L3_S29_F3342','L3_S33_F3861','L3_S29_F3430','L3_S30_F3759','L0_S23_F619','L1_S24_F1667','L3_S34_F3878','L3_S30_F3564','L3_S40_F3980','L3_S30_F3704']

#https://www.kaggle.com/dollardollar/bosch-production-line-performance/eda-of-important-features/notebook
top20 = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']

# numeric top features
top72 = ['L3_S32_F3850','L3_S33_F3855','L1_S24_F1723','L3_S33_F3865','L1_S24_F1846','L1_S24_F1695','L3_S38_F3952','L3_S29_F3407','L1_S24_F1604','L1_S24_F1632','L3_S33_F3859','L1_S24_F1565','L0_S3_F100','L0_S6_F122','L1_S24_F1778','L0_S23_F671','L3_S34_F3882','L0_S5_F116','L0_S18_F439','L0_S0_F20','L0_S1_F28','L3_S30_F3544','L3_S29_F3330','L3_S29_F3342','L1_S24_F1838','L3_S30_F3494','L3_S29_F3351','L1_S24_F1647','L3_S29_F3382','L0_S0_F18','L0_S10_F244','L3_S30_F3809','L3_S29_F3461','L0_S14_F374','L3_S30_F3804','L0_S10_F259','L3_S36_F3920','L3_S29_F3336','L0_S14_F370','L0_S11_F294','L0_S10_F219','L0_S2_F60','L2_S26_F3073','L3_S30_F3574','L3_S30_F3609','L0_S6_F132','L0_S2_F44','L0_S19_F455','L3_S30_F3769','L0_S17_F433','L3_S33_F3857','L3_S30_F3689','L3_S29_F3479','L0_S0_F2','L2_S26_F3121','L3_S30_F3754','L2_S27_F3129','L2_S27_F3140','L3_S29_F3373','L0_S13_F356','L2_S27_F3210','L2_S27_F3133','L2_S27_F3144','L0_S12_F350','L0_S0_F10','L0_S7_F138','L0_S15_F418','L3_S30_F3554','L3_S29_F3327','L0_S9_F165','L0_S15_F403','L2_S26_F3047','L2_S26_F3036','L2_S26_F3062']
#top72 = []

colsnum = list(set(cols).union(set(top72)).union(set(top20)))
usecols = ['L3_S30_F3759','L3_S30_F3809','L3_S33_F3857','L3_S30_F3769','L3_S30_F3804','L3_S30_F3754_max','L3_S30_F3754','L3_S30_F3554','L0_S0_F20','L3_S29_F3373','L3_S29_F3479_max','L3_S29_F3351','L3_S29_F3342','L3_S29_F3376','L3_S30_F3704','L3_S33_F3859','L3_S29_F3327','L3_S30_F3804_max','L3_S30_F3494','L3_S35_F3889','L3_S33_F3859_max','L3_S36_F3920','L3_S29_F3324','L3_S29_F3479','L3_S29_F3336','L3_S33_F3865','L3_S30_F3809_max','L3_S36_F3920_max','L3_S33_F3857_max','L3_S29_F3321','L0_S0_F20_max','L3_S29_F3373_max','L0_S1_F28','L3_S30_F3609','L3_S30_F3574','L0_S1_F28_max','L3_S29_F3330','L0_S0_F16','L0_S0_F18','L3_S29_F3382','L3_S30_F3574_max','L3_S30_F3604','L3_S30_F3609_max','L3_S30_F3769_max','L3_S32_F3854','L3_S30_F3494_max','L3_S29_F3354','L0_S0_F22','L3_S30_F3554_max','L3_S33_F3865_max','L3_S30_F3544','L3_S29_F3458','L0_S0_F2','L3_S29_F3342_max','L3_S29_F3370','L0_S2_F60','L3_S29_F3351_max','L1_S24_F1844','L0_S0_F18_max','L3_S29_F3436','L3_S30_F3524','L3_S29_F3430','L0_S3_F100','L0_S7_F138_max','L3_S29_F3327_max','L3_S30_F3504','L0_S12_F350','L3_S30_F3689','L1_S24_F1846','L3_S32_F3850','L0_S7_F138','L3_S30_F3689_max','L0_S5_F114','L0_S0_F2_max','L1_S24_F1571','L3_S29_F3407','L0_S2_F44_max','L0_S3_F100_max','L2_S26_F3113','L3_S29_F3382_max','L0_S2_F60_max','L3_S38_F3960','L0_S2_F44','L0_S6_F122','L0_S5_F116','L0_S12_F350_max','L3_S30_F3544_max','L3_S29_F3330_max','L0_S9_F165','L2_S26_F3117','L2_S26_F3062','L3_S38_F3956','L1_S24_F1842','L2_S26_F3121','L3_S38_F3952','L2_S26_F3073_max','L0_S0_F10','L0_S11_F294_max','L0_S5_F116_max','L2_S26_F3121_max','L1_S24_F1565','L2_S26_F3036','L1_S24_F1846_max','L0_S6_F132','L2_S26_F3047','L0_S13_F356_max','L2_S26_F3069','L0_S11_F294','L2_S26_F3073','L1_S24_F1723','L0_S17_F433_max','L0_S6_F122_max','L2_S26_F3047_max','L3_S29_F3407_max','L1_S24_F1778','L3_S32_F3850_max','L0_S10_F259','L0_S9_F165_max','L0_S13_F356','L0_S6_F132_max','L0_S10_F244','L3_S29_F3412','L1_S24_F1632','L3_S30_F3564','L3_S29_F3336_max','L1_S24_F1672','L0_S10_F259_max','L2_S26_F3036_max','L2_S26_F3062_max','L2_S27_F3210_max','L0_S0_F10_max','L2_S27_F3144_max','L1_S24_F1723_max','L0_S14_F370','L0_S18_F439_max','L2_S27_F3144','L1_S24_F1667','L0_S10_F219','L0_S14_F370_max','L0_S17_F433','L2_S27_F3133','L1_S24_F1838','L1_S24_F1647','L0_S19_F455_max','L0_S19_F455','L1_S24_F1695_max','L1_S24_F1632_max','L2_S27_F3129','L0_S23_F671','L3_S33_F3855','L0_S15_F418','L3_S29_F3461','L0_S10_F219_max','L2_S27_F3140_max','L2_S27_F3133_max','L1_S24_F1778_max','L0_S10_F244_max','L0_S23_F671_max','L1_S24_F1581','L1_S24_F1569','L3_S33_F3855_max','L0_S23_F619','L1_S24_F1604_max','L0_S15_F418_max','L0_S15_F403','L3_S29_F3467','L1_S24_F1565_max','L1_S24_F1838_max','L2_S27_F3129_max','L0_S18_F439','L2_S27_F3140','L1_S24_F1647_max','L3_S38_F3952_max','L0_S15_F403_max','L3_S41_F4016','L2_S27_F3210','L0_S14_F374_max','L3_S29_F3461_max','L3_S40_F3980','L3_S34_F3882','L3_S40_F3986','L0_S14_F374','L3_S32_F3851','L3_S29_F3317','L3_S33_F3861','L3_S43_F4080','L1_S24_F1695','L1_S24_F823','L3_S34_F3876','L3_S33_F3863','L3_S29_F3320','L3_S29_F3475','L1_S25_F2779','L3_S29_F3478','L3_S34_F3882_max','L1_S25_F1903','L1_S25_F1852','L1_S24_F1278','L3_S29_F3323','L1_S24_F675','L1_S24_F705','L3_S34_F3878','L3_S29_F3481','L1_S25_F1907','L1_S24_F710','L2_S28_F3224','L2_S28_F3228','L1_S24_F1114','L1_S25_F1912','L1_S24_F1675','L1_S25_F2141','L1_S25_F2099','L1_S25_F2229','L1_S24_F1604','L1_S25_F2958','L1_S24_F1282','L1_S25_F1981','L1_S25_F2884','L1_S25_F2104','L1_S25_F2968','L3_S29_F3326','L2_S28_F3285','L1_S25_F3013','L1_S24_F703','L1_S24_F1137','L1_S25_F2802','L1_S24_F1191','L1_S24_F1187','L2_S26_F3099','L1_S24_F1510','L2_S27_F3192','L1_S25_F1922','L1_S25_F2811','L3_S29_F3332','L1_S24_F1140','L1_S25_F1927','L1_S24_F1525','L1_S24_F1530','L1_S24_F1523','L2_S27_F3131','L1_S25_F2880','L1_S24_F1679','L1_S25_F2963','L1_S25_F2119']
colsnum  = [c for c in colsnum if c in usecols]
#category top features
colscat = ['L3_S29_F3317', 'L3_S32_F3854', 'L2_S27_F3131', 'L3_S47_F4141', 'L1_S24_F1525', 'L2_S26_F3038', 'L2_S27_F3192', 'L1_S25_F1852', 'L3_S29_F3320', 'L2_S27_F3135', 'L2_S26_F3099', 'L1_S24_F675', 'L1_S25_F2779', 'L3_S47_F4146', 'L3_S29_F3475', 'L1_S24_F1510', 'L1_S24_F710', 'L1_S24_F1530', 'L2_S26_F3042', 'L1_S24_F1675', 'L2_S26_F3082', 'L1_S25_F2519', 'L1_S25_F2802', 'L1_S25_F2958', 'L1_S25_F2963', 'L3_S29_F3478', 'L3_S29_F3323', 'L1_S24_F1584', 'L1_S25_F2806', 'L1_S25_F2523', 'L1_S24_F1114', 'L1_S24_F1191', 'L1_S25_F1981', 'L1_S24_F1187', 'L3_S32_F3851', 'L2_S26_F3045', 'L2_S27_F3138', 'L1_S25_F1907', 'L2_S26_F3085', 'L2_S26_F3049', 'L2_S27_F3142', 'L1_S25_F2496', 'L1_S25_F2099', 'L3_S29_F3481', 'L1_S25_F1912', 'L1_S25_F1903', 'L3_S47_F4156', 'L1_S24_F1136', 'L1_S24_F1592', 'L2_S28_F3224', 'L1_S24_F1679', 'L2_S26_F3053', 'L1_S24_F703', 'L1_S25_F2730', 'L1_S25_F1985', 'L1_S24_F1278', 'L1_S25_F2811', 'L3_S29_F3326', 'L3_S29_F3332', 'L1_S24_F705', 'L3_S47_F4161', 'L1_S25_F2141', 'L1_S24_F1588', 'L1_S25_F3013', 'L1_S25_F2968', 'L1_S24_F1064', 'L3_S47_F4151', 'L1_S24_F1282', 'L1_S24_F1200', 'L2_S26_F3091', 'L1_S24_F1683', 'L1_S24_F1140', 'L2_S26_F3088', 'L1_S24_F1537', 'L2_S26_F3057', 'L1_S24_F1693', 'L1_S25_F2880', 'L1_S24_F1137', 'L2_S27_F3146', 'L2_S26_F3064', 'L1_S24_F1286', 'L1_S25_F2884', 'L1_S24_F1597', 'L1_S24_F1291', 'L2_S28_F3228', 'L2_S27_F3157', 'L1_S25_F2993', 'L1_S24_F695', 'L1_S25_F2528', 'L1_S24_F1602', 'L1_S24_F1523', 'L1_S25_F1922', 'L1_S24_F1616', 'L1_S25_F2229', 'L1_S25_F2109', 'L1_S24_F1210', 'L1_S25_F2000', 'L1_S24_F1528', 'L1_S24_F1296', 'L1_S24_F1688', 'L1_S25_F2330', 'L2_S28_F3231', 'L1_S25_F1927', 'L1_S25_F2973', 'L1_S24_F1559', 'L1_S25_F2104', 'L1_S25_F2983', 'L1_S24_F1195', 'L1_S25_F2816', 'L1_S24_F1301', 'L1_S25_F2533', 'L1_S24_F1707', 'L1_S25_F2334', 'L1_S25_F2557', 'L1_S24_F1139', 'L1_S25_F2447', 'L1_S25_F2978', 'L1_S24_F823', 'L2_S28_F3285', 'L1_S25_F2119']
colscat = [c for c in colscat if c in usecols]



cols = colsnum + colscat

'''
important_features = set(['L3_S30_F3759' ,'L3_S30_F3809' ,'L3_S30_F3754' ,'L3_S29_F3351' ,'L3_S29_F3479' ,'L3_S30_F3804' ,'L3_S30_F3704' ,'L3_S30_F3804_max' ,'L3_S30_F3494' ,'L3_S36_F3920' ,'L3_S30_F3554' ,'L3_S29_F3376' ,'L3_S30_F3754_max' ,'L3_S29_F3342' ,'L3_S30_F3769' ,'L0_S0_F20' ,'L3_S35_F3889' ,'L3_S29_F3373' ,'L3_S33_F3865' ,'L3_S29_F3327' ,'L3_S30_F3609' ,'L3_S33_F3857' ,'L3_S30_F3574' ,'L3_S29_F3336' ,'L3_S32_F3854' ,'L3_S29_F3382' ,'L3_S30_F3809_max' ,'L3_S33_F3859' ,'L3_S30_F3769_max' ,'L3_S29_F3324' ,'L3_S33_F3857_max' ,'L3_S29_F3321' ,'L3_S29_F3354' ,'L0_S1_F28_max' ,'L3_S36_F3920_max' ,'L3_S30_F3574_max' ,'L3_S33_F3865_max' ,'L3_S29_F3479_max' ,'L3_S30_F3544' ,'L3_S29_F3373_max' ,'L3_S29_F3330' ,'L0_S0_F16' ,'L3_S29_F3330_max' ,'L0_S12_F350' ,'L3_S33_F3859_max' ,'L0_S0_F20_max' ,'L3_S30_F3604' ,'L3_S38_F3952' ,'L3_S29_F3430' ,'L3_S32_F3850' ,'L3_S29_F3407' ,'L3_S30_F3554_max' ,'L0_S0_F22' ,'L3_S29_F3458' ,'L3_S30_F3524' ,'L3_S29_F3351_max' ,'L3_S29_F3342_max' ,'L0_S0_F18' ,'L3_S30_F3494_max' ,'L0_S1_F28' ,'L0_S0_F18_max' ,'L3_S30_F3609_max' ,'L3_S38_F3960' ,'L3_S32_F3850_max' ,'L3_S29_F3327_max' ,'L0_S13_F356' ,'L3_S30_F3504' ,'L0_S2_F44' ,'L0_S3_F100' ,'L0_S0_F2' ,'L0_S7_F138_max' ,'L3_S29_F3382_max' ,'L0_S0_F2_max' ,'L1_S24_F1844' ,'L0_S12_F350_max' ,'L2_S26_F3073_max' ,'L3_S30_F3689_max' ,'L3_S30_F3689' ,'L3_S38_F3956' ,'L0_S3_F100_max' ,'L0_S9_F165' ,'L1_S24_F1571' ,'L3_S29_F3407_max' ,'L2_S27_F3144_max' ,'L2_S26_F3036_max' ,'L3_S29_F3436' ,'L0_S2_F60' ,'L0_S5_F116_max' ,'L2_S26_F3036' ,'L1_S24_F1846' ,'L2_S26_F3047' ,'L2_S26_F3069' ,'L1_S24_F1844_L1_S24_F1723' ,'L2_S26_F3113' ,'L2_S26_F3117' ,'L3_S29_F3412' ,'L3_S30_F3544_max' ,'L0_S6_F132_max' ,'L2_S26_F3121' ,'L1_S24_F1565' ,'L0_S6_F132' ,'L0_S0_F10_max' ,'L1_S24_F1632_L1_S24_F1844' ,'L3_S29_F3461' ,'L0_S6_F122_max' ,'L0_S17_F433_max' ,'L2_S26_F3121_max' ,'L0_S7_F138' ,'L3_S29_F3370' ,'L0_S5_F114' ,'L2_S26_F3073' ,'L2_S26_F3047_max' ,'L2_S26_F3062' ,'L3_S30_F3564' ,'L0_S10_F259' ,'L0_S2_F44_max' ,'L0_S0_F10' ,'L2_S27_F3210_max' ,'L2_S27_F3129' ,'L1_S24_F1842' ,'L0_S13_F356_max' ,'L0_S14_F370' ,'L2_S26_F3062_max' ,'L1_S24_F1778' ,'L0_S6_F122' ,'L0_S11_F294_max' ,'L0_S9_F165_max' ,'L0_S5_F116' ,'L1_S24_F1846_max' ,'L3_S29_F3336_max' ,'L0_S17_F433' ,'L0_S2_F60_max' ,'L0_S19_F455_max' ,'L0_S18_F439_max' ,'L0_S10_F259_max' ,'L2_S27_F3129_max' ,'L1_S24_F1723' ,'L3_S33_F3855' ,'L0_S11_F294' ,'L0_S18_F439' ,'L0_S14_F370_max' ,'L1_S24_F1844_L1_S24_F1632' ,'L0_S10_F219' ,'L0_S23_F671' ,'L1_S24_F1632' ,'L1_S24_F1695_L1_S24_F1723' ,'L0_S10_F219_max' ,'L1_S24_F1838' ,'L0_S10_F244' ,'L2_S27_F3133_max' ,'L1_S24_F1778_max' ,'L0_S19_F455' ,'L1_S24_F1723_L1_S24_F1844' ,'L0_S15_F418_max' ,'L1_S24_F1723_L1_S24_F1695' ,'L0_S23_F671_max' ,'L0_S15_F403_max' ,'L3_S38_F3952_max' ,'L3_S43_F4080' ,'L2_S27_F3140_max' ,'L1_S24_F1846_L1_S24_F1604' ,'L0_S15_F418' ,'L3_S29_F3467' ,'L2_S27_F3210' ,'L0_S23_F619' ,'L2_S27_F3144' ,'L1_S24_F1672_L1_S24_F1844' ,'L2_S27_F3133' ,'L1_S24_F1672' ,'L3_S33_F3855_max' ,'L2_S27_F3140' ,'L3_S29_F3461_max' ,'L1_S24_F1565_max' ,'L0_S10_F244_max' ,'L1_S24_F1569' ,'L3_S40_F3980' ,'L1_S24_F1695_L1_S24_F1846' ,'L1_S24_F1723_max' ,'L1_S24_F1846_L1_S24_F1695' ,'L0_S14_F374' ,'L1_S24_F1632_L1_S24_F1842' ,'L1_S24_F1842_L1_S24_F1723' ,'L1_S24_F1647' ,'L1_S24_F1632_L1_S24_F1667' ,'L1_S24_F1581' ,'L0_S15_F403' ,'L1_S24_F1667' ,'L1_S24_F1632_L1_S24_F1604' ,'L1_S24_F1846_L1_S24_F1667' ,'L0_S14_F374_max' ,'L1_S24_F1842_L1_S24_F1632' ,'L3_S41_F4016' ,'L3_S34_F3882_max' ,'L3_S34_F3882' ,'L3_S29_F3317' ,'L3_S32_F3851' ,'L1_S24_F1723_L1_S24_F1842' ,'L1_S24_F1838_max' ,'L1_S24_F1647_max' ,'L3_S40_F3986' ,'L1_S24_F1604_L1_S24_F1846' ,'L3_S33_F3863' ,'L3_S33_F3861' ,'L1_S24_F1844_L1_S24_F1672' ,'L1_S24_F1632_max' ,'L3_S29_F3320' ,'L1_S24_F1647_L1_S24_F1604' ,'L3_S29_F3475' ,'L1_S24_F675' ,'L3_S34_F3876' ,'L1_S24_F1667_L1_S24_F1846' ,'L3_S32_F3850Nan' ,'L1_S24_F1667_L1_S24_F1632' ,'L3_S34_F3878' ,'L1_S24_F1695_max' ,'L1_S25_F1852' ,'L1_S24_F1278' ,'L1_S25_F2779' ,'L1_S24_F1604_L1_S24_F1632' ,'L1_S25_F1903' ,'L1_S25_F2229' ,'L1_S24_F1604_L1_S24_F1647' ,'L1_S24_F1525' ,'L3_S29_F3478' ,'L1_S25_F1907' ,'L1_S24_F705' ,'L1_S24_F703' ,'L2_S28_F3228' ,'L1_S24_F1604' ,'L2_S28_F3224' ,'L1_S24_F1530' ,'L3_S29_F3323' ,'L1_S25_F1981' ,'L1_S24_F1137' ,'L3_S29_F3332' ,'L1_S24_F710' ,'L3_S29_F3481' ,'L3_S47_F4141' ,'L1_S25_F2141' ,'L1_S24_F1064' ,'L1_S24_F1191' ,'L1_S24_F1510' ,'L1_S24_F1187' ,'L1_S24_F1114' ,'L1_S24_F1675' ,'L1_S24_F1695' ,'L1_S24_F1291' ,'L1_S25_F1912' ,'L1_S25_F2958' ,'L1_S25_F2109' ,'L1_S24_F1136' ,'L1_S25_F2099' ,'L1_S25_F2802'])

colsnum = list(set(colsnum).intersection(important_features))
colscat = list(set(colscat).intersection(important_features))
top72 = list(set(top72).intersection(important_features))
cols = list(set(cols).intersection(important_features))
'''

def getdate(f):
    return '_'.join(f.split('_')[:2]) + '_D' + str(int(f.split('_')[2].replace('F',''))+1)


datefeats = list(set(pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_date.csv', nrows=1).columns.values).intersection(set([getdate(f) for f in colsnum])))
tsIndex = {}

def createTsIndex(train, features):
    for f in features:
        if '_F' not in f:
            continue
        
        d = getdate(f)
        if d not in datefeats:
            print('No date exists for {} {}'.format(f, d))
            continue

        print('Creating TS features for {} {}'.format(f,d))

        temp = train[[d,'Response']]                
        temp['ts'] = np.floor((temp[d]*1).values)

        groups = pd.groupby(temp[['ts','Response']], by=['ts']).sum().reset_index()
        groups.rename(columns={'Response': f+'_TS'}, inplace=True)
        #groups has f_TS, ts, temp has ts, d, label

        temp = pd.merge(temp, groups, on=['ts'])[[d,f+'_TS']]
        # temp has d, f_TS

        tsIndex[d] = temp.drop_duplicates().set_index(d)
        tsIndex[d].replace(0, np.nan, inplace=True)
    
    gc.collect()


def timestampFeatures(data):
    features = colsnum

    for f in features:
        if '_F' not in f:
            continue

        d = getdate(f)
        if d in tsIndex:
            temp = tsIndex[d]
            data = data.join(temp, on=[d])
            del data[d]
        gc.collect()
    gc.collect()
    return data
    






def output(modelObj):
    
    test = modelObj.test
    file = modelObj.outputFile
    fileProb = open(modelObj.prefix + '.prob', 'w')
    model = modelObj.model

    prob = model.predict_proba(modelObj.xCV.values)

    auc = metrics.roc_auc_score(modelObj.yCV.values, prob[:,1])

    modelObj.stats['auc'] = auc
    prob = prob[:,1]

    thresholds =  np.arange(99.6, 99.9, 0.025)


    bestPR = None
    bestScore =  0
    bestCF = None
    bestT = 0
    modelObj.statsFile.write('\n')
    for t in thresholds:
        temp = np.copy(prob)
        temp[np.where(prob > np.percentile(prob, t))] = 1
        temp[np.where(prob <= np.percentile(prob, t))] = 0

        score = metrics.matthews_corrcoef(modelObj.yCV.values, temp)
        pr = metrics.precision_recall_fscore_support(modelObj.yCV.values, temp, pos_label=1, average='binary')
        cf = metrics.confusion_matrix(modelObj.yCV.values, temp)
        modelObj.statsFile.write('threshold {} mcc {} auc {} pr {} tn {} fp {} fn {} tp {}\n'.format( np.percentile(prob, t), score, auc,  pr,
                                                                                                        cf[0][0], cf[0][1], cf[1][0], cf[1][1] ))

        if score > bestScore:
            bestScore = score
            bestT = np.percentile(prob, t)
            bestPR = pr
            bestCF = cf


    modelObj.statsFile.write('BEST: threshold {} mcc {} auc {} pr {} tn {}  fp {} fn {} tp {}\n'.format( bestT, bestScore, auc, bestPR,
                                                                                                        bestCF[0][0], bestCF[0][1], bestCF[1][0], bestCF[1][1]))


    test['prob'] = model.predict_proba(test[modelObj.xTrain.columns.values].values)[:,1]
    test['predict'] = test['prob'].apply(lambda x: 1 if x > bestT else 0)
    test[['label','prob']].to_csv(fileProb, index=False, delimiter=',', header=True)
    fileProb.flush()
    fileProb.close()
    temp = test[['label', 'predict']]
    file.write('Id,Response\n')
    temp.to_csv(file, index=False, delimiter=',', header=False)
    file.flush()

    return test.shape[0]





    pass




def processNumCat(outputDir):
    global cols
    global config

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    date = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_date.csv', usecols = ['Id'] + datefeats)


    cat = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_categorical.2.csv', usecols = ['Id'] + colscat)
    #train = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_categorical.2.csv', nrows=500000)

    num = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_numeric.csv', usecols=['Id', 'Response'] + colsnum)

    train = pd.merge(num, cat, on=['Id'])
    train = pd.merge(train, date, on=['Id'])

    del cat,num,date
    gc.collect()

    # create max value features
    maxVals  = train[top72].max()
    minVals = train[top72].min()
    for f in maxVals.index.values:
        train[f+'_max'] = maxVals[f] - train[f]
        if 'min' in config:
            train[f+'_min'] = train[f] - minVals[f]

    if 'nan' in config:
        train['L3_S32_F3850Nan'] = np.isnan(train['L3_S32_F3850'])
        train['L3_S33_F3859Nan'] = np.isnan(train['L3_S33_F3859'])
        train['L3_S33_F3857Nan'] = np.isnan(train['L3_S33_F3857'])
        train['L3_S33_F3865Nan'] = np.isnan(train['L3_S33_F3865'])

    if 'inter' in config:
        train['L1_S24_F1846_L1_S24_F1667']= train['L1_S24_F1846'] * train['L1_S24_F1667']
        train['L1_S24_F1846_L1_S24_F1695']= train['L1_S24_F1846'] * train['L1_S24_F1695']
        train['L1_S24_F1846_L1_S24_F1604']= train['L1_S24_F1846'] * train['L1_S24_F1604']
        train['L1_S24_F1672_L1_S24_F1844']= train['L1_S24_F1672'] * train['L1_S24_F1844']
        train['L1_S24_F1632_L1_S24_F1842']= train['L1_S24_F1632'] * train['L1_S24_F1842']
        train['L1_S24_F1632_L1_S24_F1667']= train['L1_S24_F1632'] * train['L1_S24_F1667']
        train['L1_S24_F1632_L1_S24_F1844']= train['L1_S24_F1632'] * train['L1_S24_F1844']
        train['L1_S24_F1632_L1_S24_F1604']= train['L1_S24_F1632'] * train['L1_S24_F1604']
        train['L1_S24_F1647_L1_S24_F1604']= train['L1_S24_F1647'] * train['L1_S24_F1604']
        train['L1_S24_F1842_L1_S24_F1632']= train['L1_S24_F1842'] * train['L1_S24_F1632']
        train['L1_S24_F1842_L1_S24_F1723']= train['L1_S24_F1842'] * train['L1_S24_F1723']
        train['L1_S24_F1667_L1_S24_F1846']= train['L1_S24_F1667'] * train['L1_S24_F1846']
        train['L1_S24_F1667_L1_S24_F1632']= train['L1_S24_F1667'] * train['L1_S24_F1632']
        train['L1_S24_F1695_L1_S24_F1846']= train['L1_S24_F1695'] * train['L1_S24_F1846']
        train['L1_S24_F1695_L1_S24_F1723']= train['L1_S24_F1695'] * train['L1_S24_F1723']
        train['L1_S24_F1844_L1_S24_F1672']= train['L1_S24_F1844'] * train['L1_S24_F1672']
        train['L1_S24_F1844_L1_S24_F1632']= train['L1_S24_F1844'] * train['L1_S24_F1632']
        train['L1_S24_F1844_L1_S24_F1723']= train['L1_S24_F1844'] * train['L1_S24_F1723']
        train['L1_S24_F1723_L1_S24_F1842']= train['L1_S24_F1723'] * train['L1_S24_F1842']
        train['L1_S24_F1723_L1_S24_F1695']= train['L1_S24_F1723'] * train['L1_S24_F1695']
        train['L1_S24_F1723_L1_S24_F1844']= train['L1_S24_F1723'] * train['L1_S24_F1844']
        train['L1_S24_F1604_L1_S24_F1846']= train['L1_S24_F1604'] * train['L1_S24_F1846']
        train['L1_S24_F1604_L1_S24_F1632']= train['L1_S24_F1604'] * train['L1_S24_F1632']
        train['L1_S24_F1604_L1_S24_F1647']= train['L1_S24_F1604'] * train['L1_S24_F1647']
        
    if 'hasstation' in config:
        features = train.columns.values
        lines = (set([f.split('_')[0] for f in features]))
        stations = (set([f.split('_')[1] for f in features if '_' in f]))

        for s in stations:
            fs = [f for f in features if s in f]            
            train['Has'+s] = np.max(np.isnan(train[fs]),axis=1)
            train['Has'+s] = train['Has'+s].astype(int)

    

    print(train.columns.names)

    

    cols = train.columns.values
    cols = [c for c in cols if c != 'label']

    train[colsnum] += 2
    train.fillna(0, inplace=True)

    cv = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\cv2.pkl', 'rb'))
    #cv = train.index[[i % 5 == 2 for i in train.index.tolist()]]
    #cv = random.sample(train.index.tolist(), int(train.shape[0] * (1.0/5.0)))

    cvdata = train.loc[cv]
    train= train.drop(cv)

    if 'tsfeatures' in config:
        createTsIndex(train, colsnum)
        train = timestampFeatures(train)
        cvdata = timestampFeatures(cvdata)

    del train['Id'],cvdata['Id']
    train.rename(columns={'Response': 'label'}, inplace=True)
    cvdata.rename(columns={'Response': 'label'}, inplace=True)


    date = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\test_date.csv', usecols=['Id'] + datefeats)
    cat = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\test_categorical.2.csv', usecols = ['Id'] + colscat)
    num = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\test_numeric.csv', usecols = ['Id'] + colsnum)

    test = pd.merge(num, cat, on=['Id'])
    test = pd.merge(test, date, on=['Id'])

    del cat, num, date
    test.rename(columns={'Id': 'label'}, inplace=True)


    # maxVal features
    for f in maxVals.index.values:
        test[f+'_max'] = maxVals[f] - test[f]
        if 'min' in config:
            test[f+'_min'] = test[f] - minVals[f]

    if 'nan' in config:
        test['L3_S32_F3850Nan'] = np.isnan(test['L3_S32_F3850'])
        #test['L3_S24_F1844Nan'] = np.isnan(test['L3_S24_F1844'])
        #test['L3_S24_F1846Nan'] = np.isnan(test['L3_S24_F1846'])
        test['L3_S33_F3859Nan'] = np.isnan(test['L3_S33_F3859'])
        test['L3_S33_F3857Nan'] = np.isnan(test['L3_S33_F3857'])
        test['L3_S33_F3865Nan'] = np.isnan(test['L3_S33_F3865'])

    if 'inter' in config:
        test['L1_S24_F1846_L1_S24_F1667']= test['L1_S24_F1846'] * test['L1_S24_F1667']
        test['L1_S24_F1846_L1_S24_F1695']= test['L1_S24_F1846'] * test['L1_S24_F1695']
        test['L1_S24_F1846_L1_S24_F1604']= test['L1_S24_F1846'] * test['L1_S24_F1604']
        test['L1_S24_F1672_L1_S24_F1844']= test['L1_S24_F1672'] * test['L1_S24_F1844']
        test['L1_S24_F1632_L1_S24_F1842']= test['L1_S24_F1632'] * test['L1_S24_F1842']
        test['L1_S24_F1632_L1_S24_F1667']= test['L1_S24_F1632'] * test['L1_S24_F1667']
        test['L1_S24_F1632_L1_S24_F1844']= test['L1_S24_F1632'] * test['L1_S24_F1844']
        test['L1_S24_F1632_L1_S24_F1604']= test['L1_S24_F1632'] * test['L1_S24_F1604']
        test['L1_S24_F1647_L1_S24_F1604']= test['L1_S24_F1647'] * test['L1_S24_F1604']
        test['L1_S24_F1842_L1_S24_F1632']= test['L1_S24_F1842'] * test['L1_S24_F1632']
        test['L1_S24_F1842_L1_S24_F1723']= test['L1_S24_F1842'] * test['L1_S24_F1723']
        test['L1_S24_F1667_L1_S24_F1846']= test['L1_S24_F1667'] * test['L1_S24_F1846']
        test['L1_S24_F1667_L1_S24_F1632']= test['L1_S24_F1667'] * test['L1_S24_F1632']
        test['L1_S24_F1695_L1_S24_F1846']= test['L1_S24_F1695'] * test['L1_S24_F1846']
        test['L1_S24_F1695_L1_S24_F1723']= test['L1_S24_F1695'] * test['L1_S24_F1723']
        test['L1_S24_F1844_L1_S24_F1672']= test['L1_S24_F1844'] * test['L1_S24_F1672']
        test['L1_S24_F1844_L1_S24_F1632']= test['L1_S24_F1844'] * test['L1_S24_F1632']
        test['L1_S24_F1844_L1_S24_F1723']= test['L1_S24_F1844'] * test['L1_S24_F1723']
        test['L1_S24_F1723_L1_S24_F1842']= test['L1_S24_F1723'] * test['L1_S24_F1842']
        test['L1_S24_F1723_L1_S24_F1695']= test['L1_S24_F1723'] * test['L1_S24_F1695']
        test['L1_S24_F1723_L1_S24_F1844']= test['L1_S24_F1723'] * test['L1_S24_F1844']
        test['L1_S24_F1604_L1_S24_F1846']= test['L1_S24_F1604'] * test['L1_S24_F1846']
        test['L1_S24_F1604_L1_S24_F1632']= test['L1_S24_F1604'] * test['L1_S24_F1632']
        test['L1_S24_F1604_L1_S24_F1647']= test['L1_S24_F1604'] * test['L1_S24_F1647']
   
    if 'hasstation' in config:
        features = test.columns.values
        lines = (set([f.split('_')[0] for f in features]))
        stations = (set([f.split('_')[1] for f in features if '_' in f]))

        for s in stations:
            fs = [f for f in features if s in f]            
            test['Has'+s] = np.max(np.isnan(test[fs]),axis=1)
            test['Has'+s] = test['Has'+s].astype(int)

    

    test[colsnum] += 2
    test.fillna(0, inplace=True)
    if 'tsfeatures' in config:        
        test = timestampFeatures(test)

    return train, cvdata, test





class BoschOrchestrator(Orchestrator.Orchestrator):
    def __init__(self, dataDir, outputDir, args, trainer, finalPredictFunc, resetData = False, threads = 2, debug = False, getData = None, exceptCols = [], selectCols = None):
        return super().__init__(dataDir, outputDir, args, trainer, finalPredictFunc, resetData, threads, debug, getData, exceptCols, selectCols)


def xgbCategorical():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['learning_rate'] = [0.01, 0.007, 0.005]
    #p['n_estimators'] = [200,150]
    p['max_depth'] = [7,9,12]    
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.5, 0.7, 0.9]
    p['silent'] = [False]
    p['subsample'] = [0.85, 1]
    p['base_score'] = [0.003, 0.005, 0.006]
    

    o = BoschOrchestrator('E:\Git\ML\Kaggle_Bosch\Data\DataV7Categorical\\', 
                                       r'E:\Git\ML\Kaggle_Bosch\Data\OutputXGBCategorical\\', 
                                       p, TrainModel.XGBClassifier, output, 
                                       resetData=False, threads=1, debug=True, 
                                       getData=processCategorical)

    
    o.train()

def xgbNumCat():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['learning_rate'] = [0.01]
    p['n_estimators'] = [200,300]
    p['max_depth'] = [11,12,13]    
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.5, 0.7]
    p['silent'] = [False]
    p['subsample'] = [0.85, 1]
    p['base_score'] = [0.003, 0.002]

    
    p['learning_rate'] = [0.01]
    p['n_estimators'] = [200, 175]
    p['max_depth'] = [11, 12]    
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.7, 0.8, 0.9]
    p['silent'] = [False]
    p['subsample'] = [0.85, 0.75]
    p['base_score'] = [0.003, 0.002]

    
    config.add('tsfeatures')
    
    usecols = None

    o = BoschOrchestrator(r'E:\Git\ML\Kaggle_Bosch\Data\TSFeat4\\', 
                                       r'E:\Git\ML\Kaggle_Bosch\Data\OutputXGBTSFeat\\', 
                                       p, TrainModel.XGBClassifier, output, 
                                       resetData=False, threads=1, debug=True, 
                                       getData=processNumCat, selectCols=usecols)


    
    o.train()
    '''
    config.add('nan')

    o = BoschOrchestrator('E:\Git\ML\Kaggle_Bosch\Data\DataV7NumCatMax.\\', 
                                       r'E:\Git\ML\Kaggle_Bosch\Data\OutputXGBNumCatMax.3\\', 
                                       p, TrainModel.XGBClassifier, output, 
                                       resetData=False, threads=1, debug=True, 
                                       getData=processNumCat, selectCols=None)


    
    o.train()
    '''

    
def xgb():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['learning_rate'] = [0.01, 0.007, 0.005]
    #p['n_estimators'] = [200,150]
    p['max_depth'] = [7,9,12]    
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.5, 0.7, 0.9]
    p['silent'] = [False]
    p['subsample'] = [0.85, 1]
    p['base_score'] = [0.003, 0.005, 0.006]
    

    o = BoschOrchestrator('E:\Git\ML\Kaggle_Bosch\Data\DataV7NumAllFeats\\', 
                                       r'E:\Git\ML\Kaggle_Bosch\Data\OutputXGBNumAllFeats\\', 
                                       p, TrainModel.XGBClassifier, output, 
                                       resetData=False, threads=1, debug=True, 
                                       getData=process2)

    
    o.train()



if __name__ == '__main__':
    #xgb()
    #xgbfeatureselection()
    #xgbCategorical()
    xgbNumCat()







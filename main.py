# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:20:08 2023

@author: Alan
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############################
# Figure: selfing
# Panel A: delayed dispersal level
#Data collection
data = pd.read_csv('result_phi_b_sub.csv')
db = np.array(data.loc[:, "db"])
phi = np.array(data.loc[:, "phi"])
hx_data = np.array(data.loc[:, "h_x"])
hy_data = np.array(data.loc[:, "h_y"])
#Dimensions
n_db = int(np.sqrt(len(db)))
db = db[0:n_db]
n_phi = n_db

#Phi:
Phi = np.zeros(n_phi)
for i in range(0, int(len(phi)/n_db)):
    Phi[i] = phi[i*n_db]

#hx and hy values
hx = np.zeros((n_phi, n_db))
hy = np.zeros((n_phi, n_db))

k = 0
for i in range(0, n_phi):
    for j in range(0, n_db):
        hx[i, j] = hx_data[k]
        hy[i, j] = hy_data[k]
        k += 1

#Plot:
fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
level = np.linspace(0, 0.9, 7)
cp = ax.contourf(db, Phi, hx)
fig.colorbar(cp, label="$h_x^*$ level")  
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("Selfing $\phi$", fontsize=20.0)
plt.show()

#Panel B: Reproductive value
RV_data = np.array(data.loc[:, "RV"])
RV = np.zeros((n_phi, n_db))

k = 0
for i in range(0, n_phi):
    for j in range(0, n_db):
        RV[i, j] = RV_data[k]
        k += 1

fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
levels = np.linspace(0.12, 0.17, 7)
cp = ax.contourf(db, Phi, RV,levels=levels)
fig.colorbar(cp, label="Reprod. value of a late disperser")  # Add a colorbar to a plot
plt.xlabel(' $b_y-b_x$', fontsize=20.0)
plt.show()

#Cost of delayed dispersal

cost_data = np.array(data.loc[:, "cost"])
cost = np.zeros((n_phi, n_db))

k = 0
for i in range(0, n_phi):
    for j in range(0, n_db):
        cost[i, j] = cost_data[k]
        k += 1

fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
level = np.linspace(-1.35e-5,-0.3e-5,7)
cp = ax.contourf(db, Phi, cost,levels=level)
fig.colorbar(cp, label="Cost ($\Delta_{h_x}W_{Indiv}$)")  # Add a colorbar to a plot
plt.xlabel(' $b_y-b_x$', fontsize=20.0)
plt.show()

########################################
#Figure 3: probability of establishment and delayed dispersal level
#Panel A: fecundity

#Data collection
data = pd.read_csv('result_b_sub.csv')
p2 = np.array(data.loc[:, "P2"])
h = np.array(data.loc[:, "h_x"])
db = np.array(data.loc[:, "db"])

#Dimension:
n_p = int(np.sqrt(len(p2)))
n_b = n_p
db = db[0:n_b]

#Probability of esta. and delayed dispersal level for 3 fecundity benefits:
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0, n_p):
    P1[i] = p2[10+i*len(db)]
    H1[i] = h[10+i*len(db)]
    P2[i] = p2[13+i*len(db)]
    H2[i] = h[13+i*len(db)]
    P3[i] = p2[17+i*len(db)]
    H3[i] = h[17+i*len(db)]

#Plot:
fig, ax = plt.subplots()
color=np.linspace(1,0,len(P1))
plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
plt.xlim(0.05,0.15)
plt.ylim(0.07,0.37)
plt.xlabel("Probability of establishment",fontsize=20)
plt.ylabel("$h_x^*$ level",fontsize=20)

#Panel B: survival
#Data collection
data = pd.read_csv('result_s_sub.csv')
p2 = np.array(data.loc[:, "P2"])
h = np.array(data.loc[:, "h_x"])
ds = np.array(data.loc[:, "ds"])

#Dimension:
n_p = int(np.sqrt(len(p2)))
n_s = n_p
ds = ds[0:n_s]
#Probability of esta. and delayed dispersal level for 3 survival benefits:
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0, n_p):
    P1[i] = p2[10+i*len(ds)]
    H1[i] = h[10+i*len(ds)]
    P2[i] = p2[12+i*len(ds)]
    H2[i] = h[12+i*len(ds)]
    P3[i] = p2[15+i*len(ds)]
    H3[i] = h[15+i*len(ds)]


#Plot:
fig, ax = plt.subplots()
color=np.linspace(1,0.9,len(P3))
plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
plt.xlim(0.05,0.27)
plt.ylim(0,0.6)
plt.xlabel("Probability of establishment",fontsize=20)
plt.ylabel("$h_x^*$ level",fontsize=20)

#Panel C: subordinate's survival
#Data colelction
data = pd.read_csv('result_say_sub.csv')
p2 = np.array(data.loc[:, "P2"])

h = np.array(data.loc[:, "h_x"])
ds = np.array(data.loc[:, "ds"])
n_p = int(np.sqrt(len(p2)))
n_s = n_p
ds = ds[0:n_s]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0, n_p):
    P1[i] = p2[5+i*len(ds)]
    H1[i] = h[5+i*len(ds)]
    P2[i] = p2[10+i*len(ds)]
    H2[i] = h[10+i*len(ds)]
    P3[i] = p2[15+i*len(ds)]
    H3[i] = h[15+i*len(ds)]

#Plot:
fig, ax = plt.subplots()
color=np.linspace(1,0,len(P1))
plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
plt.xlabel("Probability of establishment",fontsize=20)
plt.ylabel("$h_x^*$ level",fontsize=20)
plt.xlim(0.12,0.16)
plt.ylim(0,1.05)


fig, ax = plt.subplots()
hb = np.linspace(0,5)
plt.plot(hb,np.exp(-hb),color="black")
plt.xlabel("$h_xb_x$",fontsize=20)
plt.ylabel("$T_x'$",fontsize=20)

###########################
#Figure 4: breeder-subordinate conflicts
#Panel A: fecundity
#Ai, ii: no group augmentation
#Data collection
data_sub = pd.read_csv('result_b_sub_3.csv')
data_bre = pd.read_csv('result_b_bre_3.csv')

db = np.array(data_sub.loc[:, "db"])
b = np.array(data_sub.loc[:, "b"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_db = int(len(db)/3)
n_b = 3
db = db[0:n_db]
B = np.zeros(3)

for i in range(0, n_b):
    B[i] = b[i*n_db]
    
#Computation of the data
hx_data_sub_b = hx_data_sub[n_db:2*n_db]
hx_data_bre_b = hx_data_bre[n_db:2*n_db]
#Difference between breeder and subordinate
hx_data_diff_b1 = hx_data_bre[0:n_db]-hx_data_sub[0:n_db]
hx_data_diff_b2 = hx_data_bre[n_db:2*n_db]-hx_data_sub[n_db:2*n_db]
hx_data_diff_b3 = hx_data_bre[-n_db:]-hx_data_sub[-n_db:]

#Thresholds:
th_sub = db[len(db)-sum(hx_data_sub_b>0)-1]
th_bre = db[len(db)-sum(hx_data_bre_b>0)-1]

#Plot of the level
fig, ax = plt.subplots()
plt.plot(db, hx_data_sub_b, color="black")
plt.plot(db, hx_data_bre_b, color="grey")
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])

#Plot of the difference, i.e. the conflict
fig, ax = plt.subplots()
plt.plot(db, hx_data_diff_b1, color="black", linestyle="dashed")
plt.plot(db, hx_data_diff_b2, color="black")
plt.plot(db, hx_data_diff_b3, color="black", linestyle="dotted")
plt.legend(["$b_x=4.0$", "$b_x=5.0$", "$b_x=6.0$"])
plt.ylim([0,1])

plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)

#Aiii and iv: with group augmentation
#Data:
data_sub = pd.read_csv('result_b_GA_sub_3.csv')
data_bre = pd.read_csv('result_b_GA_bre_3.csv')
db = np.array(data_sub.loc[:, "db"])
b = np.array(data_sub.loc[:, "b"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_db = int(len(db)/3)
n_b = 3
db = db[0:n_db]
B = np.zeros(3)

for i in range(0, n_b):
    B[i] = b[i*n_db]

#Delay dispersal level
hx_data_sub_b = hx_data_sub[n_db:2*n_db]
hx_data_bre_b = hx_data_bre[n_db:2*n_db]
#Threshold
th_sub=db[len(db)-sum(hx_data_sub_b>0)-1]
th_bre=db[len(db)-sum(hx_data_bre_b>0)-1]
#Conflict
hx_data_diff_b1 = hx_data_bre[0:n_db]-hx_data_sub[0:n_db]
hx_data_diff_b2 = hx_data_bre[n_db:2*n_db]-hx_data_sub[n_db:2*n_db]
hx_data_diff_b3 = hx_data_bre[-n_db:]-hx_data_sub[-n_db:]

#Plot
fig, ax = plt.subplots()
plt.plot(db, hx_data_sub_b, color="black")
plt.plot(db, hx_data_bre_b, color="grey")
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Conflict plot
fig, ax = plt.subplots()
plt.plot(db, hx_data_diff_b1, color="black", linestyle="dashed")
plt.plot(db, hx_data_diff_b2, color="black")
plt.plot(db, hx_data_diff_b3, color="black", linestyle="dotted")
plt.legend(["$b_x=4.0$", "$b_x=5.0$", "$b_x=6.0$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)
#######################################


# Panel B: survival benefits
#Bi and ii: no group augmentation
#Data collection
data_sub = pd.read_csv('result_s_sub_3.csv')
data_bre = pd.read_csv('result_s_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
hx_data_sub_s = hx_data_sub[n_ds:2*n_ds]
hx_data_bre_s = hx_data_bre[n_ds:2*n_ds]

#Threshold
th_sub=ds[len(ds)-sum(hx_data_sub_s>0)-1]
th_bre=ds[len(ds)-sum(hx_data_bre_s>0)-1]
#Conflict
hx_data_diff_s1 = hx_data_bre[0:n_ds]-hx_data_sub[0:n_ds]
hx_data_diff_s2 = hx_data_bre[n_ds:2*n_ds]-hx_data_sub[n_ds:2*n_ds]
hx_data_diff_s3 = hx_data_bre[-n_ds:]-hx_data_sub[-n_ds:]

#Plot
fig, ax = plt.subplots()
plt.plot(ds, hx_data_sub_s, color="black")
plt.plot(ds, hx_data_bre_s, color="grey")
plt.xlabel('$s_y-s_x$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)

#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(ds, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, hx_data_diff_s2, color="black")
plt.plot(ds, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_x=0.5$", "$s_x=0.6$", "$s_x=0.7$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_y-s_x$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)


#Panel Biii, iv: with group augmentation
#Data collection 
data_sub = pd.read_csv('result_s_GA_sub_3.csv')
data_bre = pd.read_csv('result_s_GA_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
hx_data_sub_s = hx_data_sub[n_ds:2*n_ds]
hx_data_bre_s = hx_data_bre[n_ds:2*n_ds]
#Threshold
th_sub=ds[len(ds)-sum(hx_data_sub_s>0)-1]
th_bre=ds[len(ds)-sum(hx_data_bre_s>0)-1]
#Conflict
hx_data_diff_s1 = hx_data_bre[0:n_ds]-hx_data_sub[0:n_ds]
hx_data_diff_s2 = hx_data_bre[n_ds:2*n_ds]-hx_data_sub[n_ds:2*n_ds]
hx_data_diff_s3 = hx_data_bre[-n_ds:]-hx_data_sub[-n_ds:]

#Plot
fig, ax = plt.subplots()
plt.plot(ds, hx_data_sub_s, color="black")
plt.plot(ds, hx_data_bre_s, color="grey")
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_y-s_x$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(ds, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, hx_data_diff_s2, color="black")
plt.plot(ds, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_x=0.5$", "$s_x=0.6$", "$s_x=0.7$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_y-s_x$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)



#Panel C: group augmentation level: i and ii
#Data collection
data_sub = pd.read_csv('result_say_sub_3.csv')
data_bre = pd.read_csv('result_say_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
hx_data_sub_s = hx_data_sub[n_ds:2*n_ds]
hx_data_bre_s = hx_data_bre[n_ds:2*n_ds]
#Conflict
hx_data_diff_s1 = hx_data_bre[0:n_ds]-hx_data_sub[0:n_ds]
hx_data_diff_s2 = hx_data_bre[n_ds:2*n_ds]-hx_data_sub[n_ds:2*n_ds]
hx_data_diff_s3 = hx_data_bre[-n_ds:]-hx_data_sub[-n_ds:]

#Plot 
fig, ax = plt.subplots()
plt.plot(ds, hx_data_sub_s, color="black")
plt.plot(ds, hx_data_bre_s, color="grey")
plt.xlabel('$s_{a_y}-s_{a_x}$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.xaxis.set_ticks(np.arange(0, max(ds)+0.01, 0.05))
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])
fig, ax = plt.subplots()

#Plot of the conflict
plt.plot(ds, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, hx_data_diff_s2, color="black")
plt.plot(ds, hx_data_diff_s3, color="black", linestyle="dotted")
ax.xaxis.set_ticks(np.arange(0, max(ds)+0.01, 0.05))
plt.ylim([0,1])
plt.legend(["$s_{a_x}=0.5$", "$s_{a_x}=0.6$", "$s_{a_x}=0.7$"])
plt.xlabel('$s_{a_y}-s_{a_x}$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)



##################################
# Supplementary data
#Level of delay dispersal when group bring benefits only to the survival of subordinates: s_u<s_ax=s_ay, by=bx, sy=sx
#and conflict over delay dispersal
#Data collection 
data_sub = pd.read_csv('result_sa_sub_3.csv')
data_bre = pd.read_csv('result_sa_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
hx_data_sub_s = hx_data_sub[n_ds:2*n_ds]
hx_data_bre_s = hx_data_bre[n_ds:2*n_ds]
#Conflict
hx_data_diff_s1 = hx_data_bre[0:n_ds]-hx_data_sub[0:n_ds]
hx_data_diff_s2 = hx_data_bre[n_ds:2*n_ds]-hx_data_sub[n_ds:2*n_ds]
hx_data_diff_s3 = hx_data_bre[-n_ds:]-hx_data_sub[-n_ds:]

#Plot
fig, ax = plt.subplots()
plt.plot(ds, hx_data_sub_s, color="black")
plt.plot(ds, hx_data_bre_s, color="grey")
plt.xlabel('$b_y-b_x$', fontsize=20.0)
plt.ylabel("$h_x$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Conflict over delay dispersal
fig, ax = plt.subplots()
plt.plot(ds, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, hx_data_diff_s2, color="black")
plt.plot(ds, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_{a_x}=0.5$", "$s_{a_x}=0.6$", "$s_{a_x}=0.7$"])
plt.xlabel('$s_y-s_x$', fontsize=20.0)
plt.ylabel("$h_x(bre)-h_x(sub)$", fontsize=20.0)


##############
#Effect of varying the selfing rate phi over dispersal level and conflict with group augmentation and fecundity benefits
#Data collection
data_sub = pd.read_csv('result_phi_b_GA_sub_3.csv')
data_bre = pd.read_csv('result_phi_b_GA_bre_3.csv')
phi = np.array(data_sub.loc[:, "phi"])
db = np.array(data_sub.loc[:, "db"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_phi = int(len(phi)/3)
n_db = 3
Phi = phi[0:n_phi]
#Data computation
hx_data_sub_b = hx_data_sub[n_phi:2*n_phi]
hx_data_bre_b = hx_data_bre[n_phi:2*n_phi]
#Conflict
hx_data_diff_bGA1 = hx_data_bre[0:n_phi]-hx_data_sub[0:n_phi]
hx_data_diff_bGA2 = hx_data_bre[n_phi:2*n_phi]-hx_data_sub[n_phi:2*n_phi]
hx_data_diff_bGA3 = hx_data_bre[-n_phi:]-hx_data_sub[-n_phi:]

#pàPlot
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_sub_b, color="black")
plt.plot(Phi, hx_data_bre_b, color="grey")
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_diff_bGA1, color="black", linestyle="dashed")
plt.plot(Phi, hx_data_diff_bGA2, color="black")
plt.plot(Phi, hx_data_diff_bGA3, color="black", linestyle="dotted")
plt.legend(["$b_y-b_x=2$", "$b_y-b_x=3$", "$b_y-b_x=4$"])
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)

###############################
#Effect of varying the selfing rate phi over dispersal level and conflict with group augmentation and survival benefits
#Data collection
data_sub = pd.read_csv('result_phi_s_GA_sub_3.csv')
data_bre = pd.read_csv('result_phi_s_GA_bre_3.csv')
phi = np.array(data_sub.loc[:, "phi"])
ds = np.array(data_sub.loc[:, "ds"])
hx_data_sub = np.array(data_sub.loc[:, "h_x"])
hx_data_bre = np.array(data_bre.loc[:, "h_x"])

#Dimension
n_phi = int(len(phi)/3)
n_ds = 3
Phi = phi[0:n_phi]
db = np.zeros(3)
#Computation of the data
for i in range(0, n_ds):
    ds[i] = s[i*n_phi]
hx_data_sub_s = hx_data_sub[0:n_phi]
hx_data_bre_s = hx_data_bre[0:n_phi]
#Conflict
hx_data_diff_s1 = hx_data_bre[0:n_phi]-hx_data_sub[0:n_phi]
hx_data_diff_s2 = hx_data_bre[n_phi:2*n_phi]-hx_data_sub[n_phi:2*n_phi]
hx_data_diff_s3 = hx_data_bre[-n_phi:]-hx_data_sub[-n_phi:]

#Plot
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_sub_s, color="black")
plt.plot(Phi, hx_data_bre_s, color="grey")
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_x^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(Phi, hx_data_diff_s2, color="black")
plt.plot(Phi, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_y-s_x=0.2$", "$s_y-s_x=0.3$", "$s_y-s_x=0.4$"])
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_x^*(bre)-h_x^*(sub)$", fontsize=20.0)

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
data = pd.read_csv("result_phi_n_sub.csv")
dn = np.array(data.loc[:, "dn"])
phi = np.array(data.loc[:, "phi"])
h0_data = np.array(data.loc[:, "h_0"])
h1_data = np.array(data.loc[:, "h_1"])
#Dimensions
size = int(np.sqrt(len(dn)))
phi = phi[0:size]

#Phi:
d_n = np.zeros(size)
for i in range(0, size):
    d_n[i] = dn[i*size]
#hx and hy values
h0 = np.zeros((size,size))
h1 = np.zeros((size,size))

k = 0
for i in range(0, size):
    for j in range(0, size):
        h0[i, j] = h0_data[k]
        h1[i, j] = h1_data[k]
        k += 1

#Plot:
fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
level = np.linspace(0, 0.9, 7)
cp = ax.contourf(d_n, phi, h0)
fig.colorbar(cp, label="$h_0^*$ level")  
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("Selfing $\phi$", fontsize=20.0)
plt.show()

#Panel B: Reproductive value
GS_data = np.array(data.loc[:, "GS"])
GS = np.zeros((size, size))

k = 0
for i in range(0, size):
    for j in range(0, size):
        GS[i, j] = GS_data[k]
        k += 1

fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
levels = np.linspace(0.12, 0.17, 7)
cp = ax.contourf(d_n, phi, GS,levels=levels)
fig.colorbar(cp, label="Reprod. value of a late disperser")  # Add a colorbar to a plot
plt.xlabel(' $n_1-n_0$', fontsize=20.0)
plt.show()

#Cost of delayed dispersal

cost_data = np.array(data.loc[:, "cost"])
cost = np.zeros((size,size))

k = 0
for i in range(0, size):
    for j in range(0, size):
        cost[i, j] = cost_data[k]
        k += 1

fig, ax = plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize=15)
level = np.linspace(-1.35e-5,-0.3e-5,7)
cp = ax.contourf(d_n, phi, cost,levels=level)
fig.colorbar(cp, label="Cost ($\Delta_{h_0}W_{Actor}$)")  # Add a colorbar to a plot
plt.xlabel(' $n_1-n_0$', fontsize=20.0)
plt.show()

########################################
#Figure 3: probability of establishment and delayed dispersal level
#Panel A: fecundity

#Data collection
# data = pd.read_csv('result_n_sub.csv')
# p2 = np.array(data.loc[:, "P2"])
# h = np.array(data.loc[:, "h_0"])
# dn = np.array(data.loc[:, "dn"])

# #Dimension:
# size = int(np.sqrt(len(p2)))
# dn = dn[0:size]
# print(dn)
# #Probability of esta. and delayed dispersal level for 3 fecundity benefits:
# P1 = np.zeros(size)
# H1 = np.zeros(size)
# P2 = np.zeros(size)
# H2 = np.zeros(size)
# P3 = np.zeros(size)
# H3 = np.zeros(size)
# for i in range(0, size):
#     P1[i] = p2[2+i*size]
#     H1[i] = h[2+i*size]
#     P2[i] = p2[3+i*size]
#     H2[i] = h[3+i*size]
#     P3[i] = p2[4+i*size]
#     H3[i] = h[4+i*size]

# #Plot:
# fig, ax = plt.subplots()
# color=np.linspace(1,0,size)
# plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
# plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
# plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
# plt.xlim(0.05,0.15)
# plt.ylim(0.07,0.37)
# plt.xlabel("Probability of establishment",fontsize=20)
# plt.ylabel("$h_0^*$ level",fontsize=20)

#Panel B: survival
#Data collection
data = pd.read_csv('result_s_sub.csv')
p2 = np.array(data.loc[:, "P2"])
h = np.array(data.loc[:, "h_0"])
ds = np.array(data.loc[:, "ds"])
#Dimension:
size = int(np.sqrt(len(p2)))
ds = ds[0:size]
#Probability of esta. and delayed dispersal level for 3 survival benefits:
P1 = np.zeros(size)
H1 = np.zeros(size)
P2 = np.zeros(size)
H2 = np.zeros(size)
P3 = np.zeros(size)
H3 = np.zeros(size)

for i in range(0, size):
  
    P1[i] = p2[10+i*len(ds)]
    H1[i] = h[10+i*len(ds)]
    P2[i] = p2[12+i*len(ds)]
    H2[i] = h[12+i*len(ds)]
    P3[i] = p2[15+i*len(ds)]
    H3[i] = h[15+i*len(ds)]

#Plot:
fig, ax = plt.subplots()
color=np.linspace(1,0.9,size)
plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
plt.xlim(0.05,0.27)
plt.ylim(0,0.6)
plt.xlabel("Probability of establishment",fontsize=20)
plt.ylabel("$h_0^*$ level",fontsize=20)

#Panel C: subordinate's survival
#Data colelction
data = pd.read_csv('result_sa1_sub.csv')
p2 = np.array(data.loc[:, "P2"])

h = np.array(data.loc[:, "h_0"])
ds = np.array(data.loc[:, "ds"])
size = int(np.sqrt(len(p2)))

ds = ds[0:size]
P1 = np.zeros(size)
H1 = np.zeros(size)
P2 = np.zeros(size)
H2 = np.zeros(size)
P3 = np.zeros(size)
H3 = np.zeros(size)
for i in range(0, size):
    P1[i] = p2[5+i*size]
    H1[i] = h[5+i*size]
    P2[i] = p2[10+i*size]
    H2[i] = h[10+i*size]
    P3[i] = p2[15+i*size]
    H3[i] = h[15+i*size]
#5, 10, 15

#Plot:
fig, ax = plt.subplots()
color=np.linspace(1,0,size)
plt.scatter(P3, H3,s=150,c=color,cmap="RdBu")
plt.scatter(P2, H2,s=70,c=color,cmap="RdBu")
plt.scatter(P1, H1,s=20,c=color,cmap="RdBu")
plt.xlabel("Probability of establishment",fontsize=20)
plt.ylabel("$h_0^*$ level",fontsize=20)
plt.xlim(0.12,0.16)
plt.ylim(0,1.05)


fig, ax = plt.subplots()
hb = np.linspace(0,5)
plt.plot(hb,np.exp(-hb),color="black")
plt.xlabel("Group size after early dispersal: $h_0n_0$",fontsize=20)
plt.ylabel("$T_0'$",fontsize=20)

###########################
#Figure 4: breeder-subordinate conflicts
#Panel A: fecundity
#Ai, ii: no group augmentation
#Data collection
data_sub = pd.read_csv('result_n_sub_3.csv')
data_bre = pd.read_csv('result_n_bre_3.csv')

dn = np.array(data_sub.loc[:, "dn"])
n = np.array(data_sub.loc[:, "n"])
hx_data_sub = np.array(data_sub.loc[:, "h_0"])
hx_data_bre = np.array(data_bre.loc[:, "h_0"])

#Dimension
n_dn = int(len(dn)/3)
n_n = 3
dn = dn[0:n_dn]
N = np.zeros(3)

for i in range(0, n_n):
    N[i] = n[i*n_dn]
    
#Computation of the data
hx_data_sub_n = hx_data_sub[n_dn:2*n_dn]
hx_data_bre_n = hx_data_bre[n_dn:2*n_dn]
#Difference between breeder and subordinate
hx_data_diff_n1 = hx_data_bre[0:n_dn]-hx_data_sub[0:n_dn]
hx_data_diff_n2 = hx_data_bre[n_dn:2*n_dn]-hx_data_sub[n_dn:2*n_dn]
hx_data_diff_n3 = hx_data_bre[-n_dn:]-hx_data_sub[-n_dn:]

#Thresholds:
th_sub = dn[len(dn)-sum(hx_data_sub_n>0)-1]
th_bre = dn[len(dn)-sum(hx_data_bre_n>0)-1]

#Plot of the level
fig, ax = plt.subplots()
plt.plot(dn, hx_data_sub_n, color="black")
plt.plot(dn, hx_data_bre_n, color="grey")
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])

#Plot of the difference, i.e. the conflict
fig, ax = plt.subplots()
plt.plot(dn, hx_data_diff_n1, color="black", linestyle="dashed")
plt.plot(dn, hx_data_diff_n2, color="black")
plt.plot(dn, hx_data_diff_n3, color="black", linestyle="dotted")
plt.legend(["$n_0=4.0$", "$n_0=5.0$", "$n_0=6.0$"])
plt.ylim([0,1])

plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)

#Aiii and iv: with group augmentation
#Data:
data_sub = pd.read_csv('result_n_GS_sub_3.csv')
data_bre = pd.read_csv('result_n_GS_bre_3.csv')
dn = np.array(data_sub.loc[:, "dn"])
n = np.array(data_sub.loc[:, "n"])
h0_data_sub = np.array(data_sub.loc[:, "h_0"])
h0_data_bre = np.array(data_bre.loc[:, "h_0"])

#Dimension
n_dn = int(len(dn)/3)
n_n = 3
dn = dn[0:n_dn]
N = np.zeros(3)

for i in range(0, n_n):
    N[i] = n[i*n_dn]

#Delay dispersal level
h0_data_sub_n = h0_data_sub[n_dn:2*n_dn]
h0_data_bre_n = h0_data_bre[n_dn:2*n_dn]
#Threshold
th_sub=dn[len(dn)-sum(h0_data_sub_n>0)-1]
th_bre=dn[len(dn)-sum(h0_data_bre_n>0)-1]
#Conflict
hb_data_diff_n1 = h0_data_bre[0:n_dn]-h0_data_sub[0:n_dn]
hb_data_diff_n2 = h0_data_bre[n_dn:2*n_dn]-h0_data_sub[n_dn:2*n_dn]
hb_data_diff_n3 = h0_data_bre[-n_dn:]-h0_data_sub[-n_dn:]

#Plot
fig, ax = plt.subplots()
plt.plot(dn, h0_data_sub_n, color="black")
plt.plot(dn, h0_data_bre_n, color="grey")
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Conflict plot
fig, ax = plt.subplots()
plt.plot(dn, hb_data_diff_n1, color="black", linestyle="dashed")
plt.plot(dn, hb_data_diff_n2, color="black")
plt.plot(dn, hb_data_diff_n3, color="black", linestyle="dotted")
plt.legend(["$n_0=4.0$", "$n_0=5.0$", "$n_0=6.0$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)
#######################################


# Panel B: survival benefits
#Bi and ii: no group size effects
#Data collection
data_sub = pd.read_csv('result_s_sub_3.csv')
data_bre = pd.read_csv('result_s_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
h0_data_sub = np.array(data_sub.loc[:, "h_0"])
h0_data_bre = np.array(data_bre.loc[:, "h_0"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
h0_data_sub_s = h0_data_sub[n_ds:2*n_ds]
h0_data_bre_s = h0_data_bre[n_ds:2*n_ds]

#Threshold
th_sub=ds[len(ds)-sum(h0_data_sub_s>0)-1]
th_bre=ds[len(ds)-sum(h0_data_bre_s>0)-1]
#Conflict
h0_data_diff_s1 = h0_data_bre[0:n_ds]-h0_data_sub[0:n_ds]
h0_data_diff_s2 = h0_data_bre[n_ds:2*n_ds]-h0_data_sub[n_ds:2*n_ds]
h0_data_diff_s3 = h0_data_bre[-n_ds:]-h0_data_sub[-n_ds:]

#Plot
fig, ax = plt.subplots()
plt.plot(ds, h0_data_sub_s, color="black")
plt.plot(ds, h0_data_bre_s, color="grey")
plt.xlabel('$s_{b_1}-s_{b_0}$', fontsize=20.0)
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)

#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(ds, h0_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, h0_data_diff_s2, color="black")
plt.plot(ds, h0_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_{b_0}=0.5$", "$s_{b_0}=0.6$", "$s_{b_0}=0.7$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_{b_1}-s_{b_0}$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)


#Panel Biii, iv: with group augmentation
#Data collection 
data_sub = pd.read_csv('result_s_GS_sub_3.csv')
data_bre = pd.read_csv('result_s_GS_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
h0_data_sub = np.array(data_sub.loc[:, "h_0"])
h0_data_bre = np.array(data_bre.loc[:, "h_0"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
h0_data_sub_s = h0_data_sub[n_ds:2*n_ds]
h0_data_bre_s = h0_data_bre[n_ds:2*n_ds]
#Threshold
th_sub=ds[len(ds)-sum(h0_data_sub_s>0)-1]
th_bre=ds[len(ds)-sum(h0_data_bre_s>0)-1]
#Conflict
h0_data_diff_s1 = h0_data_bre[0:n_ds]-h0_data_sub[0:n_ds]
h0_data_diff_s2 = h0_data_bre[n_ds:2*n_ds]-h0_data_sub[n_ds:2*n_ds]
h0_data_diff_s3 = h0_data_bre[-n_ds:]-h0_data_sub[-n_ds:]

#Plot
fig, ax = plt.subplots()
plt.plot(ds, h0_data_sub_s, color="black")
plt.plot(ds, h0_data_bre_s, color="grey")
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_{b_1}-s_{b_0}$', fontsize=20.0)
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(ds, h0_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, h0_data_diff_s2, color="black")
plt.plot(ds, h0_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_{b_0}=0.5$", "$s_{b_0}=0.6$", "$s_{b_0}=0.7$"])
plt.ylim([0,1])
plt.axvline(x=th_sub,color="brown",linewidth=0.5)
plt.axvline(x=th_bre,color="orange",linewidth=0.5)
plt.xlabel('$s_{b_1}-s_{b_0}$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)



#Panel C: group augmentation level: i and ii
#Data collection
data_sub = pd.read_csv('result_sa1_sub_3.csv')
data_bre = pd.read_csv('result_sa1_bre_3.csv')
ds = np.array(data_sub.loc[:, "ds"])
s = np.array(data_sub.loc[:, "s"])
h0_data_sub = np.array(data_sub.loc[:, "h_0"])
h0_data_bre = np.array(data_bre.loc[:, "h_0"])

#Dimension
n_ds = int(len(ds)/3)
n_s = 3
ds = ds[0:n_ds]
S = np.zeros(3)

#Data computation
for i in range(0, n_s):
    S[i] = s[i*n_ds]
h0_data_sub_s = h0_data_sub[n_ds:2*n_ds]
h0_data_bre_s = h0_data_bre[n_ds:2*n_ds]
#Conflict
h0_data_diff_s1 = h0_data_bre[0:n_ds]-h0_data_sub[0:n_ds]
h0_data_diff_s2 = h0_data_bre[n_ds:2*n_ds]-h0_data_sub[n_ds:2*n_ds]
h0_data_diff_s3 = h0_data_bre[-n_ds:]-h0_data_sub[-n_ds:]

#Plot 
fig, ax = plt.subplots()
plt.plot(ds, h0_data_sub_s, color="black")
plt.plot(ds, h0_data_bre_s, color="grey")
plt.xlabel('$s_{a_1}-s_{a_0}$', fontsize=20.0)
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.xaxis.set_ticks(np.arange(0, max(ds)+0.01, 0.05))
ax.legend(["sub ctrl", "breeder ctrl"])
plt.ylim([0,1])
fig, ax = plt.subplots()

#Plot of the conflict
plt.plot(ds, h0_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, h0_data_diff_s2, color="black")
plt.plot(ds, h0_data_diff_s3, color="black", linestyle="dotted")
ax.xaxis.set_ticks(np.arange(0, max(ds)+0.01, 0.05))
plt.ylim([0,1])
plt.legend(["$s_{a_0}=0.5$", "$s_{a_0}=0.6$", "$s_{a_0}=0.7$"])
plt.xlabel('$s_{a_1}-s_{a_0}$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)



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
plt.xlabel('$n_1-n_0$', fontsize=20.0)
plt.ylabel("$h_0$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])

#Conflict over delay dispersal
fig, ax = plt.subplots()
plt.plot(ds, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(ds, hx_data_diff_s2, color="black")
plt.plot(ds, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_{a_0}=0.5$", "$s_{a_0}=0.6$", "$s_{a_0}=0.7$"])
plt.xlabel('$s_{b_1}-s_{b_0}$', fontsize=20.0)
plt.ylabel("$h_0(bre)-h_0(sub)$", fontsize=20.0)


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
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_diff_bGA1, color="black", linestyle="dashed")
plt.plot(Phi, hx_data_diff_bGA2, color="black")
plt.plot(Phi, hx_data_diff_bGA3, color="black", linestyle="dotted")
plt.legend(["$n_1-n_0=2$", "$n_1-n_0=3$", "$n_1-n_0=4$"])
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)

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
plt.ylabel("$h_0^*$ level", fontsize=20.0)
ax.legend(["sub ctrl", "breeder ctrl"])
#Plot of the conflict
fig, ax = plt.subplots()
plt.plot(Phi, hx_data_diff_s1, color="black", linestyle="dashed")
plt.plot(Phi, hx_data_diff_s2, color="black")
plt.plot(Phi, hx_data_diff_s3, color="black", linestyle="dotted")
plt.legend(["$s_{b_1}-s_{b_0}=0.2$", "$s_{b_1}-s_{b_0}=0.3$", "$s_{b_1}-s_{b_0}=0.4$"])
plt.xlabel('$\phi$', fontsize=20.0)
plt.ylabel("$h_0^*(bre)-h_0^*(sub)$", fontsize=20.0)

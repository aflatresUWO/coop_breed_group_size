# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:43:15 2022

@author: aflatres
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
######################
#First figure
#Total level of help in groups with fecundity benefits
#Data:
data = pd.read_csv('result_p_b_all.csv')

db = np.array(data.loc[:,"db"])
b = np.array(data.loc[:,"b"])
hx_data = np.array(data.loc[:,"h_x"])
hy_data = np.array(data.loc[:,"h_y"])


n_db = int(np.sqrt(len(db)))
n_b = n_db
db = db[0:n_db]

B = np.zeros(n_b)

P = np.zeros(n_b)
for i in range(0,n_b):
    B[i] = b[i*n_db]

hxbx = np.zeros(n_b*n_db)
hyby = np.zeros(n_b*n_db)

k = 0
for i in range(0,n_b):
    for j in range(0,n_db):
        hxbx[k] = hx_data[k]*B[i]
        hyby[k] = hy_data[k]*(B[i]+db[j])
        k+=1
        
fig, ax=plt.subplots()
ax.scatter(hxbx,hyby)
ax.plot(hxbx,hxbx,color = "red")
plt.xlabel('$h_xb_x$',fontsize = 20.0)
plt.ylabel("$h_yb_y$",fontsize = 20.0)

############################################

#Total level of help in groups with survival benefits
data = pd.read_csv('result_p_s.csv')

ds = np.array(data.loc[:,"ds"])
s = np.array(data.loc[:,"s"])

hx_data = np.array(data.loc[:,"h_x"])
hy_data = np.array(data.loc[:,"h_y"])

n_ds = int(np.sqrt(len(ds)))
ds = ds[0:n_ds]
n_s = n_ds


S = np.zeros(n_s)
P = np.zeros(n_s)
for i in range(0,int(len(s)/n_ds)):
    S[i] = s[i*n_ds]
hx = np.zeros((n_s,n_ds))
hy = np.zeros((n_s,n_ds))

k = 0
for i in range(0,n_s):
    for j in range(0,n_ds):
        hx[i,j] = hx_data[k]
        hy[i,j] = hy_data[k]
        k+=1
        
fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)

ax.scatter(hx_data,hy_data)
ax.plot(hx_data,hx_data,color="red")
plt.xlabel('$h_x$',fontsize = 20.0)
plt.ylabel("$h_y$",fontsize = 20.0)
########################################
#Second figure: group augmentation
#Level of help with group augmentation vs none
data = pd.read_csv('result_p_b_all_2.csv')

db = np.array(data.loc[:,"db"])
b = np.array(data.loc[:,"b"])
hx_data = np.array(data.loc[:,"h_x"])
hy_data = np.array(data.loc[:,"h_y"])


n_db = int(np.sqrt(len(db)))
n_b = n_db
db = db[0:n_db]

B = np.zeros(n_b)
P = np.zeros(n_b)
for i in range(0,int(len(b)/n_db)):
    B[i] = b[i*n_db]
    
hx = np.zeros(n_b*n_db)
hy = np.zeros(n_b*n_db)

data = pd.read_csv('result_p_GA_2.csv')
hx_GA_data = np.array(data.loc[:,"h_x"])
hy_GA_data = np.array(data.loc[:,"h_y"])
    
hx_diff = np.zeros((n_b,n_db))
hy_diff = np.zeros((n_b,n_db))

k = 0
for i in range(0,n_b):
    for j in range(0,n_db):
        hx_diff[i,j] = hx_GA_data[k]-hx_data[k]
        hy_diff[i,j] = hy_GA_data[k]-hy_data[k]
        k+=1

fig, ax=plt.subplots()

plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
level = np.linspace(0,0.5,7)


cp=ax.contourf(db,B,hx_diff)
level = np.linspace(0,0,1)
ax.contour(db,B,hx_diff,levels = level,colors = "red")
fig.colorbar(cp,label = "$\Delta h_x$")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("Fecundity $b_x$",fontsize = 20.0)
plt.show()



############################
#Third figure: selfing 
#Phi and fecundity benefits
data = pd.read_csv('result_phi_b_cost_2.csv')

db = np.array(data.loc[:,"db"])
phi = np.array(data.loc[:,"phi"])
hx_data = np.array(data.loc[:,"h_x"])
hy_data = np.array(data.loc[:,"h_y"])

n_db = int(np.sqrt(len(db)))
db = db[0:n_db]
n_phi = n_db


Phi = np.zeros(n_phi)

for i in range(0,int(len(phi)/n_db)):
    Phi[i] = phi[i*n_db]
    
hx = np.zeros((n_phi,n_db))
hy = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        hx[i,j] = hx_data[k]
        hy[i,j] = hy_data[k]
        k+=1

fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
level = np.linspace(0,0.25,7)

cp=ax.contourf(db,Phi,hx)
level = np.linspace(0,0,1)
#ax.contour(db,Phi,hx,levels = level,colors = "red")
fig.colorbar(cp,label = "$h_x$")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.ylabel("Selfing $\phi$",fontsize = 20.0)
#plt.title("Contour plot of ESS $h_x$ vs $\Delta b$ and probability of establishment",fontsize = 15.0)
plt.show()
######################################
#Phi and GA
GA_data = np.array(data.loc[:,"GA"])
GA = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        GA[i,j] = GA_data[k]
        k+=1

fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
level = np.linspace(0.085,0.092,7)

cp=ax.contourf(db,Phi,GA)
fig.colorbar(cp,label = "GA")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.show()
#########################
#Phi and cost
cost_data = np.array(data.loc[:,"cost"])
cost = np.zeros((n_phi,n_db))

k = 0
for i in range(0,n_phi):
    for j in range(0,n_db):
        cost[i,j] = cost_data[k]
        k+=1

fig, ax=plt.subplots()
plt.rc('font', size=15)
plt.rc('axes', labelsize = 15)
level = np.linspace(0.085,0.092,7)

cp=ax.contourf(db,Phi,cost)
fig.colorbar(cp,label = "Cost")#Add a colorbar to a plot
plt.xlabel('Fecundity benefits $\Delta b$',fontsize = 20.0)
plt.show()
#########################
#Fourth figure: probability of establishment
#Probability of establishment and hx when varying fecundity
fig, ax=plt.subplots()
data = pd.read_csv('result_p_b_all.csv')
p2 = np.array(data.loc[:,"P2"])
h= np.array(data.loc[:,"h_x"])
db= np.array(data.loc[:,"db"])
n_p = int(np.sqrt(len(p2)))
n_b = n_p
db = db[0:n_b]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,n_p):
    P1[i] = p2[7+i*len(db)]
    H1[i] = h[7+i*len(db)]
    P2[i] = p2[12+i*len(db)]
    H2[i] = h[12+i*len(db)]
    P3[i] = p2[17+i*len(db)]
    H3[i] = h[17+i*len(db)]
    

P1 = P1[H1>0]
H1 = H1[H1>0]
P2 = P2[H2>0]
H2 = H2[H2>0]
P3 = P3[H3>0]
H3 = H3[H3>0]

plt.scatter(P3,H3)
plt.scatter(P2,H2)
plt.scatter(P1,H1)
plt.legend(["$\Delta b=1.68$","$\Delta b=1.16$","$\Delta b=0.63$"],fontsize =13.0, bbox_to_anchor=[0.36,1])
########################
#Probability of establishment and hx when varying survival
fig, ax=plt.subplots()
data = pd.read_csv('result_p_s.csv')
p2 = np.array(data.loc[:,"P2"])
h= np.array(data.loc[:,"h_x"])
ds= np.array(data.loc[:,"ds"])

n_p = int(np.sqrt(len(ds)))
n_s = n_p
ds = ds[0:n_s]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,n_p):
    P1[i] = p2[7+i*n_s]
    H1[i] = h[7+i*n_s]
    P2[i] = p2[12+i*n_s]
    H2[i] = h[12+i*n_s]
    P3[i] = p2[17+i*n_s]
    H3[i] = h[17+i*n_s]
    

P1 = P1[H1>0]
H1 = H1[H1>0]
P2 = P2[H2>0]
H2 = H2[H2>0]
P3 = P3[H3>0]
H3 = H3[H3>0]

plt.scatter(P3,H3)
plt.scatter(P2,H2)
plt.scatter(P1,H1)
plt.legend(["$\Delta s=0.17$","$\Delta s=0.12$","$\Delta s=0.06$"],fontsize =13.0, bbox_to_anchor=[0.64,0.66])


#########################
#Probability of establishment and hx when varying fecundity
fig, ax=plt.subplots()
data = pd.read_csv('result_p_b_all.csv')
p2 = np.array(data.loc[:,"P2"])
h= np.array(data.loc[:,"h_x"])
db= np.array(data.loc[:,"db"])

n_db = int(np.sqrt(len(db)))
n_p = n_db
db = db[0:n_db]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,n_p):
    P1[i] = p2[i+7*n_db]
    H1[i] = h[i+7*n_b]
    P2[i] = p2[i+12*n_db]
    H2[i] = h[i+12*n_db]
    P3[i] = p2[i+17*n_db]
    H3[i] = h[i+17*n_db]
    

P1 = P1[H1>0]
H1 = H1[H1>0]
P2 = P2[H2>0]
H2 = H2[H2>0]
P3 = P3[H3>0]
H3 = H3[H3>0]

plt.scatter(P3,H3,color="black")
plt.scatter(P2,H2,color="red")
plt.scatter(P1,H1,color = "orange")
plt.xlim(0.07,0.145)
plt.legend(["$b_x=6.58$","$ b_x=5.53$","$b_x=4.47$"],fontsize =13.0, bbox_to_anchor=[0.66,0.98])
########################

#########################
#Probability of establishment and hx when varying survival
fig, ax=plt.subplots()
data = pd.read_csv('result_p_s.csv')
p2 = np.array(data.loc[:,"P2"])
h= np.array(data.loc[:,"h_x"])
ds= np.array(data.loc[:,"ds"])
n_p = int(np.sqrt(len(ds)))
n_s = n_p

ds = db[0:n_s]
P1 = np.zeros(n_p)
H1 = np.zeros(n_p)
P2 = np.zeros(n_p)
H2 = np.zeros(n_p)
P3 = np.zeros(n_p)
H3 = np.zeros(n_p)
for i in range(0,n_p):
    P1[i] = p2[i+7*n_s]
    H1[i] = h[i+7*n_s]
    P2[i] = p2[i+12*n_s]
    H2[i] = h[i+12*n_s]
    P3[i] = p2[i+17*n_s]
    H3[i] = h[i+17*n_s]
    
P1 = P1[H1>0]
H1 = H1[H1>0]
P2 = P2[H2>0]
H2 = H2[H2>0]
P3 = P3[H3>0]
H3 = H3[H3>0]

plt.scatter(P3,H3,color="black")
plt.scatter(P2,H2,color="red")
plt.scatter(P1,H1,color="orange")
plt.xlim(0.025,0.145)

plt.legend(["$s_x=0.47$","$ s_x=0.39$","$s_x=0.31$"],fontsize =13.0, bbox_to_anchor=[0.66,0.98])


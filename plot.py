import matplotlib.pyplot as plt
import wandb
import torch 

import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')


def plot_rate_distorsion(bpp_res, psnr_res):


    legenda = {}
    legenda["gain"] = {}
    legenda["our"] = {}
    legenda["EVC"] = {}
    legenda["Balle"] = {}



    legenda["gain"]["colore"] = [palette[8],'-']
    legenda["gain"]["legends"] = "Balle18 + Cui21"
    legenda["gain"]["symbols"] = ["*"]*6
    legenda["gain"]["markersize"] = [5]*6



    legenda["Balle"]["colore"] = [palette[0],'-']
    legenda["Balle"]["legends"] = "Balle18"
    legenda["Balle"]["symbols"] = ["*"]*6
    legenda["Balle"]["markersize"] = [5]*6


    legenda["EVC"]["colore"] = [palette[8],'-']
    legenda["EVC"]["legends"] = "EVC"
    legenda["EVC"]["symbols"] = ["*"]*6
    legenda["EVC"]["markersize"] = [5]*6

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "Balle18 + Stanh"
    legenda["our"]["symbols"] = ["*"]*len(psnr_res["our"])
    legenda["our"]["markersize"] = [8]*len(psnr_res["our"])

    
    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(bpp_res.keys()) #[base our]


    list_names = list(bpp_res.keys())

    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        symbols = legenda[type_name]["symbols"]
        markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]

    
        bpp = torch.tensor(bpp).cpu().numpy()
        psnr = torch.tensor(psnr).cpu().numpy()
        #plt.plot(bpp,psnr,"-", label =  leg ,color = colore, markersize=7)
        #for x, y, marker, markersize_t in zip(bpp, psnr, symbols, markersize):
        


        if "our" in type_name:
            plt.plot(bpp,psnr,"-", label =  leg ,color = colore, markersize=7)
            #plt.plot(bpp[1], psnr[1], marker="*", markersize=15, color = colore)
            #plt.plot(bpp[4], psnr[4], marker="*", markersize=15, color = colore)
            plt.plot(bpp[-1], psnr[-1], marker="*", markersize=15, color = colore)
            plt.plot(bpp[0], psnr[0], marker="o", markersize=7, color = colore)
            plt.plot(bpp[2], psnr[2], marker="o", markersize=7, color = colore)
            plt.plot(bpp[3], psnr[3], marker="o", markersize=7, color = colore)     
            #plt.plot(bpp[5], psnr[5], marker="o", markersize=7, color = colore)
            #plt.plot(bpp[6], psnr[6], marker="o", markersize=7, color = colore)  
        elif "Balle" in type_name:
            plt.plot(bpp,psnr,"-.", label =  leg ,color = colore, markersize=7)
            for i in range(bpp.shape[0]):
                plt.plot(bpp[i], psnr[i], marker="*", markersize=15, color = colore)
        else:
            plt.plot(bpp,psnr,"-.", label =  leg ,color = colore, markersize=7)
            for i in range(bpp.shape[0]):
                plt.plot(bpp[i], psnr[i], marker="o", markersize=7, color = colore)



        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]

    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    plt.savefig("variable_rate_resultBalle.png")
    plt.close()  
    print("FINITO")


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), np.sort(PSNR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), np.sort(PSNR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff

import numpy as np
import scipy.interpolate
import numpy as np
def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), np.sort(lR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), np.sort(lR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff




def main():

    psnr_res = {}
    bpp_res = {}


    #psnr_res["our"] = [28.65,29.34,31.35,32.72,33.5428,34.499,35.836,37.236]
    #bpp_res["our"] = [0.12,0.1501,0.26,0.34,0.4,0.55,0.67,0.84]
    #bpp_res["EVC"] =  [0.332920792, 0.505808708, 0.738309958, 0.987386125]
    #psnr_res["EVC"] = [32.46911329,34.42935488,36.34560738,37.75338558]
    bpp_res["gain"] =  [0.1635,0.2367,0.339,0.4731,0.6324,0.8124]
    psnr_res["gain"] = [29.303,30.842,32.377,33.908,35.306,36.502]

    psnr_res["our"] = [26.255,27.198,29.355,30.342,30.937]
    bpp_res["our"] = [0.09,0.12,0.21,0.27,0.325]

    bpp_res["gain"] =  [0.098, 0.204,0.3364]
    psnr_res["gain"] =  [25.864,27.256,29.189]

    bpp_res["Balle"] =  [ 0.13129340277777776,
      0.20889282226562503,
      0.3198581271701389,]
    psnr_res["Balle"] =  [ 27.58153679333392,
      29.19669412138477,
      30.972168705499175,]

    plot_rate_distorsion(bpp_res,psnr_res)
    r1 = [0.332920792, 0.505808708, 0.738309958, 0.987386125]
    psnr1 = [32.46911329,34.42935488,36.34560738,37.75338558]
    psnr2 = [31.35,32.72,33.5428,34.499,35.836,37.236]
    r2 = [0.26,0.34,0.4,0.55,0.67,0.84]
    c = BD_RATE(r1,psnr1,r2, psnr2)
    print(c)

if __name__ == "__main__":

  
    main()
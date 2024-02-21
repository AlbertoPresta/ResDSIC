import matplotlib.pyplot as plt
import wandb
import torch 

import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')


def plot_rate_distorsion(bpp_res, psnr_res, entropy_estimation = "model"):


    legenda = {}
    legenda["gain"] = {}
    legenda["our"] = {}
    legenda["EVC"] = {}



    legenda["gain"]["colore"] = [palette[0],'-']
    legenda["gain"]["legends"] = "Gain"
    legenda["gain"]["symbols"] = ["*"]*6
    legenda["gain"]["markersize"] = [5]*6



    legenda["EVC"]["colore"] = [palette[8],'-']
    legenda["EVC"]["legends"] = "EVC"
    legenda["EVC"]["symbols"] = ["*"]*6
    legenda["EVC"]["markersize"] = [5]*6

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "proposed"
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
            plt.plot(bpp[1], psnr[1], marker="*", markersize=15, color = colore)
            plt.plot(bpp[4], psnr[4], marker="*", markersize=15, color = colore)
            plt.plot(bpp[-1], psnr[-1], marker="*", markersize=15, color = colore)
            plt.plot(bpp[0], psnr[0], marker="o", markersize=7, color = colore)
            plt.plot(bpp[2], psnr[2], marker="o", markersize=7, color = colore)
            plt.plot(bpp[3], psnr[3], marker="o", markersize=7, color = colore)     
            plt.plot(bpp[5], psnr[5], marker="o", markersize=7, color = colore)
            plt.plot(bpp[6], psnr[6], marker="o", markersize=7, color = colore)  
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
    plt.savefig("variable_rate_result.pdf")
    plt.close()  
    print("FINITO")
    






def main():

    psnr_res = {}
    bpp_res = {}


    psnr_res["our"] = [28.65,29.34,31.35,32.72,33.5428,34.499,35.836,37.236]
    bpp_res["our"] = [0.12,0.1501,0.26,0.34,0.4,0.55,0.67,0.84]


    bpp_res["EVC"] =  [0.332920792, 0.505808708, 0.738309958, 0.987386125]
    psnr_res["EVC"] = [32.46911329,34.42935488,36.34560738,37.75338558]

    bpp_res["gain"] =  [0.1635,0.2367,0.339,0.4731,0.6324,0.8124]
    psnr_res["gain"] = [29.303,30.842,32.377,33.908,35.306,36.502]

    plot_rate_distorsion(bpp_res,psnr_res)


if __name__ == "__main__":

  
    main()


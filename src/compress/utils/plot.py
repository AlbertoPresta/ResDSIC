import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
#rc('font', family='Times New Roman')
import matplotlib.pyplot as plt
import wandb
import torch 
def plot_rate_distorsion(bpp_res, psnr_res,epoch, entropy_estimation = "model"):


    legenda = {}
    legenda["base"] = {}
    legenda["our"] = {}



    legenda["base"]["colore"] = [palette[0],'-']
    legenda["base"]["legends"] = "reference"
    legenda["base"]["symbols"] = ["*"]*6
    legenda["base"]["markersize"] = [5]*6

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "proposed"
    legenda["our"]["symbols"] = ["*"]*len(psnr_res["our"])
    legenda["our"]["markersize"] = [5]*len(psnr_res["our"])

    
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
        plt.plot(bpp,psnr,"-", label =  leg ,color = colore, markersize=7)
        #for x, y, marker, markersize_t in zip(bpp, psnr, symbols, markersize):
        plt.plot(bpp, psnr, marker="o", markersize=7)
                



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
    wandb.log({entropy_estimation:epoch,
              entropy_estimation + "/rate distorsion trade-off": wandb.Image(plt)})
    plt.close()  
    print("FINITO")
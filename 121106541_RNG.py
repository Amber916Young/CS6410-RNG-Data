from ast import parse
from email import parser
import string
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def LOG(p):
    return [1 / (-np.log(1 - p)) * ((p ** 1) / 1)]

def logseries(k):
    pmf = -p**k/(k*np.log(1-p))
    return pmf

def getCDFPMFtext(dataset,pmfPro):
    res = []
    for (i1, i2) in zip(dataset,pmfPro):
        dictr = str(i1)+","+str(i2)
        res.append(dictr)
    np.savetxt('LOG_CDF_x_Px.txt', res, delimiter=',',fmt='%s')   
    
def Logarithmic_distribution(seed,number,p):
    rng = np.random.RandomState(seed)
    x = rng.logseries(p, number) 
    fig, ax = plt.subplots(figsize=(8,4))
    org_data = x
    org_data = [logseries(d) for d in x]
    np.savetxt('LOG_sequence.txt',x, fmt="%d", delimiter=",")   # use exponential notation
    n_bins = 10
#     n_bins = list(set(x))
#     n_bins.append(0)
#     n_bins = sorted(n_bins)
#     n_bins.append(n_bins[-1] + 1)
    # plot the cumulative histogram
    count, bins, ignored = plt.hist(x, n_bins,color='r',histtype='step', cumulative=True,density=True, label='Empirical')
#     logList = [1 / (-math.log(1 - p)) * ((p ** 1) / 1)]

    y = logseries(bins) * count.max()/logseries(bins).max() 
    y = y.cumsum()
    y /= y[-1]
    ax.plot(bins,y, ls='--',color='k', linewidth=1.5, label='Expected')

    #Overlay a reversed cumulative histogram.
    ax.hist(x, n_bins, ls="-.",color='g',histtype='step', cumulative=-1,  density=True, label='Reversed emp*')

    getCDFPMFtext(bins,org_data)
   
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('LOG Cumulative step histograms')
    ax.set_xlabel('LOG RNG')
    ax.set_ylabel('LOG CDF')
    filename = "LOG_p_"+str(p)+"_number_"+str(number)+".pdf"
    plt.savefig(filename,bbox_inches="tight")
    filename2 = "LOG_p_"+str(p)+"_number_"+str(number)+".png"
    plt.savefig(filename2,bbox_inches="tight")
    plt.show()
    
    

    fig, ax = plt.subplots(figsize=(8,2))
    ax.boxplot(x)
    ax.set_title('LOG Box Plot')
    ax.set_xlabel('series')
    

    filename = "LOG_Box_Plot_p_"+str(p)+"_number_"+str(number)+".pdf"
    plt.savefig(filename,bbox_inches="tight")
    filename22 = "LOG_Box_Plot_p_"+str(p)+"_number_"+str(number)+".png"
    plt.savefig(filename22,bbox_inches="tight")
    plt.show()



def inv_cdf_kumaraswamy( y,a,b):
    return (1 - (1 - y) ** (1 / b)) ** (1 / a)

def kumaraswamyPDF(dataset,a , b):
     
    pdfPro = (a*b*(dataset**(a-1))*(1.0-dataset**a)**(b-1))
    res = []
    for (i1, i2) in zip(dataset,pdfPro):
        dictr = str(i1)+","+str(i2)
        res.append(dictr)
    np.savetxt('KUM_CDF_x_Px.txt', res, delimiter=',',fmt='%s')   

def cdf_kumaraswamy(dataset,a,b):
    return 1-((1-dataset**a)**(b))


def kumaraswamy(a, b,seed,N):
    X1 = []
    fig, ax = plt.subplots(figsize=(8,2))
    n_bins = N//2
    rng = np.random.RandomState(seed)
    X1 = rng.uniform( 0.0,1.0,size=N )
    X1 = sorted(X1)
    np.savetxt('KUM_sequence.txt', X1, delimiter=',',fmt='%s')   # use exponential notation
    X1 = np.array(X1)

    gen = inv_cdf_kumaraswamy(X1,a,b)
    bins_value, bins, patches = ax.hist(gen, n_bins,color="r", density=True, histtype="step", cumulative=True,
                                          label="Empirical")
    
    theList = cdf_kumaraswamy(bins,a,b)
    ax.plot(bins,theList,ls=":",color="k",linewidth=1.5 ,label="Theoretical")
    
    
    ax.hist(gen, n_bins,ls="--",color="c", density=True, histtype="step", cumulative=-1,
                                          label="Reversed emp*")
    
    kumaraswamyPDF(X1,a,b)
    
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('KUM Cumulative Histogram')
    ax.set_xlabel('KUM')
    ax.set_ylabel('KUM CDF')     
    
    filename = "KUM_"+str(a)+"_"+str(b)+".pdf"
    plt.savefig(filename,bbox_inches="tight")
    filename2 = "KUM_"+str(a)+"_"+str(b)+".png"
    plt.savefig(filename2,bbox_inches="tight")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8,2))
    ax.boxplot(gen)
    ax.set_title("KUM BoxPlot")
    ax.set_xlabel('series')
    
    filename = "KUM_Box_Plot_"+str(a)+"_"+str(b)+".pdf"
    plt.savefig(filename,bbox_inches="tight")
    filename22 = "KUM_Box_Plot_"+str(a)+"_"+str(b)+".png"
    plt.savefig(filename22,bbox_inches="tight")
    
    plt.show()


# python3 [filename] 
# [n] [seed] [distribution name] [par1] [par2]
parser = argparse.ArgumentParser(description='Two Distributions')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('n',type=int,help="number") 
required.add_argument('seed',type=int,help="seed") 
required.add_argument('r_type',type=str,help="distribution name") 
required.add_argument('par1', help="par1") 
required.add_argument('--par2',type=int,help="par2", default=3,required=False) 
allType = ["KUM", "LOG","log","kum"]
args = parser.parse_args()
if __name__=="__main__":
    seed = args.seed
    number= args.n
    r_type = args.r_type
    if(r_type == "log" or r_type == "LOG"):
        p = float(args.par1)
        Logarithmic_distribution(seed,number,p)
    else:
        a = int(args.par1)
        b= int(args.par2)
        kumaraswamy(a,b,seed,number)

    if r_type not in ["KUM", "LOG","log","kum"]:
        sys.exit(f"Distribution must be in the list {allType}.")

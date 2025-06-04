import os
import pandas as pd
import matplotlib.pyplot as plt

class DataLogger():
    def __init__(self,gameName):
        self.gameName=gameName
        self.basePath="./rewards/"
        self.path=self.basePath+gameName+".csv"
        self.lastEpoch=0
        self.columns=["Epoch","Average reward","Min reward","Max reward","Cumulative Reward"]
         
        if not os.path.exists(self.basePath):
            os.makedirs(self.basePath)

        if not os.path.exists(self.path):
            self.createFile()
        else:
            #Leggo il file per capire da dove devo partire
            df=pd.read_csv(self.path)
            self.lastEpoch=len(df)

    

    def log(self,data):
        path=self.path
        self.lastEpoch=self.lastEpoch+1
        epoch=[self.lastEpoch]
        minReward=[data['minReward']]
        maxReward=[data['maxReward']]
        totalReward=[data['totalReward']]
        averageReward=[data['reward']]
        data = {
            "Epoch": epoch, 
            "Average reward": averageReward, 
            "Min reward": minReward, 
            "Max reward": maxReward,
            "Cumulative Reward":totalReward
            }
        
        df=pd.DataFrame(data,columns=self.columns)
        df.to_csv(path,mode='a',index=False,header=False)


    def createFile(self): 
        data = {
            "Epoch": [], 
            "Average reward": [], 
            "Min reward": [], 
            "Max reward": [],
            "Cumulative Reward":[]
            }
        
        df=pd.DataFrame(data,columns=self.columns)
        df.to_csv(self.path,index=False, header=True)

    def showGraphCumulativeReward(self):
        data=pd.read_csv(self.path)
        plt.figure(figsize=(10, 6))
        plt.plot(data['Epoch'], data['Cumulative Reward'], label='Sum Rewards', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Sum Rewards')
        plt.title('Sum Rewards per Epoch: '+self.gameName)
        plt.legend()
        plt.grid(True)
        plt.show()
        return
    
    def showGraphAverageReward(self):
        data=pd.read_csv(self.path)
        plt.figure(figsize=(10, 6))
        plt.plot(data['Epoch'], data['Average reward'], label='Average Reward', color='blue', linewidth=2)
        plt.fill_between(data['Epoch'], data['Min reward'], data['Max reward'], color='blue', alpha=0.2, label='Range (Min-Max)')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Average Reward with Min-Max Range: '+self.gameName)
        plt.legend()
        plt.grid(True)
        plt.show()
        return



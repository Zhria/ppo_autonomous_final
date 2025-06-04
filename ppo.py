import tensorflow as tf
import numpy as np
from ac_ppo import ActorCriticPPO
import os

class PPO(tf.keras.Model):
    def __init__(self,env,weightsPath,recorder,dataLogger,gameName,training=True):
        """
        HYPER PARAMETERS        
        """
        
        #Training
        self.gamma=0.99   # discount factor
        self.lam=0.97   # lambda factor for GAE
        self.nSteps=4096 #Steps per epoch
        self.nMiniBatches=12 #Num mini batches per grad. descent step
        self.nBatchTrain=self.nSteps//self.nMiniBatches
        self.nEpochs=35# Number of epochs to run
        self.saveWeightsFreq=5 #Save every n epochs

        self.lastDoneEnv=False #Used to avoid Warning message Warning: early reset ignored        
        self.lastObs=env.reset() ##Used to avoid Warning message Warning: early reset ignored

        

        #Policy
        self.clip_grads=0.5 # value for gradient clipping
        self.epsilon=0.2
        self.valueCoefficient= 0.5  # Value coef for backprop
        self.entropyCoeffiecient=0.020
        self.learningRate=3e-4
        self.trainIterations=5 # pi and v update iterations for backprop
        self.target_kl=0.03 # target kl divergence
        """
        END HYPER PARAMETERS
        """
        self.training=training #If training is False the model is used for evaluation
        self.env=env
        self.nActions=env.action_space.n
        self.recorder=None
        self.dataLogger=None
        if training:
            self.recorder=recorder
            self.dataLogger=dataLogger
        self.gameName=gameName
        
        

        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learningRate, clipnorm=self.clip_grads)
        self.weightOptimizerPath="./weights/"+gameName+"/opt"
        

        if weightsPath is None:
            if not os.path.exists("./weights/"+gameName):
                os.makedirs("./weights/"+gameName)
            self.weightsActorPath="./weights/"+gameName+"/actor.keras"
            self.weightsCriticPath="./weights/"+gameName+"/critic.keras"

        else:
            self.weightsActorPath=weightsPath+"/actor.keras"
            self.weightsCriticPath=weightsPath+"/critic.keras"
        self.ActorCritic=ActorCriticPPO(env,self.optimizer,weigthsActor=self.weightsActorPath, weigthsCritic=self.weightsCriticPath)

    def getActionAndLogProb(self, batch_obs, deterministic=False):
        logits,values=self.ActorCritic.forward(batch_obs)
        if deterministic:
            actions=tf.argmax(tf.nn.softmax(logits), axis=-1)
        else:
            actions=tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        logp_t=self.logp(logits,actions)
        return np.squeeze(actions),np.squeeze(logp_t),np.squeeze(values), logits


    def logp(self, logits, actions):
        """
            Returns:
            logp based on the action drawn from prob-distribution indexes in the logp_all with one_hot
        """
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(actions, depth= self.nActions)
        logp = tf.reduce_sum( one_hot * logp_all, axis= -1)
        return logp


    def entropy(self,logits):
        """
            Entropy term for more randomness which means more exploration \n
            Based on OpenAI Baselines implementation
        """
        a0 = logits - tf.reduce_max(logits, axis= -1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis= -1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis= -1)
        return -tf.reduce_mean(entropy)*self.entropyCoeffiecient
    
    def rollout(self):
        batch_obs = []
        batch_rews = []
        batch_dones = []
        batch_actions = []
        batch_values = []
        batch_logp = []
        frames=[]
        if self.lastDoneEnv == True:
            obs=self.env.reset()
        else:
            obs=self.lastObs
        i=0
        last_done=False
        for _ in range(self.nSteps):
            action,logp_t,value,_=self.getActionAndLogProb(obs)
            obs,rew,done,_=self.env.step(action)
            batch_obs.append(obs)
            frames.append(obs)
            batch_actions.append(action)
            batch_values.append(value)
            batch_logp.append(logp_t)
            batch_rews.append(rew)
            batch_dones.append(done)
            self.lastDoneEnv=done
            self.lastObs=obs
            if done:
                obs=self.env.reset()
                self.recorder.saveRecord(frames,i,False)
                frames=[]
                i+=1

        _,_,last_values,_=self.getActionAndLogProb(self.lastObs)
        print("Reward rollout: ",np.sum(batch_rews))
        last_done=batch_dones[-1] #Last done value
        #Calc advantages and returns
        returns = np.zeros_like(batch_rews)
        advs = np.zeros_like(batch_rews)
        last_gae_lam = 0
        for t in reversed(range(self.nSteps)):
            if t == self.nSteps - 1:
                next_non_terminal = 1.0 - last_done #True=1.0, False=0.0
                next_values = last_values
            else:
                next_non_terminal = 1.0 - batch_dones[t + 1]
                next_values = batch_values[t + 1]
            
            delta = batch_rews[t] + self.gamma * next_values * next_non_terminal - batch_values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            
        returns = advs + batch_values                   
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)      #Normalization

        if np.sum(batch_rews)==0:
            print("No rewards in this epoch")
        
        return tf.convert_to_tensor(batch_obs) ,tf.convert_to_tensor(batch_actions) ,tf.reshape(tf.convert_to_tensor(batch_logp,dtype=tf.float32),[-1]), tf.convert_to_tensor(returns), tf.convert_to_tensor(advs)
    

    def losses(self, obs, logp_old, actions, advs, returns):
        _,_,values,logits=self.getActionAndLogProb(obs)
        logp=self.logp(logits,actions)
        logp=tf.reshape(tf.convert_to_tensor(logp,dtype=tf.float32),shape=[-1])
        ratio = tf.exp(logp - logp_old)
        approx_kl = tf.reduce_mean(logp_old - logp)                                     # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        clipped_ratio =tf.clip_by_value(ratio, clip_value_min=1.0-self.epsilon, clip_value_max=1.0+self.epsilon)
        entropy_loss=self.entropy(logits)
        clipped_loss = tf.minimum(ratio * advs, clipped_ratio * advs)

        policy_loss=-tf.reduce_mean(clipped_loss)
        value_loss=tf.reduce_mean(tf.square(returns - values)) * self.valueCoefficient
        total_loss=policy_loss+value_loss+entropy_loss
        return total_loss, approx_kl
    

    def train(self, obs, actions, advs, returns, logp):
        inds = np.arange(self.nSteps)

        for i in range(self.trainIterations):
            total_loss, approx_kl = self.update_loop( obs, actions, advs, returns, logp,inds) 

            if approx_kl > self.target_kl:
                print("Early stopping at step %d due to reaching max kl." %i)
                break

        return total_loss   



    def update_loop(self, obs, actions, advs, returns, logp_t, inds):
        np.random.shuffle(inds)
        means = []

        for start in range(0, self.nSteps, self.nBatchTrain):

            end = start + self.nBatchTrain
            obs_batch = obs[start:end]
            actions_batch = actions[start:end]
            advs_batch = advs[start:end]
            returns_batch = returns[start:end]
            logp_t_batch = logp_t[start:end]
           
            total_loss,approx_kl = self.applyGradients(obs_batch,actions_batch,logp_t_batch,advs_batch,returns_batch)

            means.append([total_loss, approx_kl])
        
        means = np.asarray(means)
        means = np.mean(means, axis= 0)

        return means[0], means[1] #Total_loss and approx_kl

    #train_one_step
    def applyGradients(self, obs, actions, logp_old,advs, returns):
        with tf.GradientTape() as tape:
            total_loss,approx_kl= self.losses(obs, logp_old, actions, advs, returns)
        trainable_variables = self.ActorCritic.get_trainable_variables()                            # take all trainable variables into account
        grads = tape.gradient(total_loss, trainable_variables)              

        self.optimizer.apply_gradients(zip(grads, trainable_variables))                 # Backprop gradients through network

        return total_loss, approx_kl


    def evaluate_model(self,episode=10,deterministic=True):
        total_rewards = []
        total_steps_per_episode=[]
        for i in range(episode):
            frames=[]

            state = self.env.reset()
            done = False
            cumulative_reward = 0
            while not done:
                action, _,_,_= self.getActionAndLogProb(state,deterministic=deterministic)
                next_state, reward, done, _ =self.env.step(action)
                state=next_state
                frames.append(state)
                cumulative_reward += reward
            total_rewards.append(cumulative_reward)
            total_steps_per_episode.append(len(frames))
            if self.training:
                self.recorder.saveRecord(frames,i,True)
            
            print("Episode reward:", cumulative_reward, "Episode steps:", len(frames))

        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        print(f"Min Reward: {np.min(total_rewards):.2f}")
        print(f"Max Reward: {np.max(total_rewards):.2f}")
        print(f"Cumulative Reward: {np.sum(total_rewards):.2f}")
        print("Average steps per episode: ",np.mean(total_steps_per_episode))
        print("Total Steps: ",np.sum(total_steps_per_episode))
        if self.training:
            self.dataLogger.log({
                "reward":np.mean(total_rewards),
                "minReward":np.min(total_rewards),
                "maxReward":np.max(total_rewards),
                "totalReward":np.sum(total_rewards)
            })

        return np.mean(total_rewards),np.min(total_rewards),np.max(total_rewards),np.sum(total_rewards)
    

    def save_model(self,epoch):
        if epoch % self.saveWeightsFreq == 0:
            self.ActorCritic.save()
            print("Model saved at epoch: ",epoch)


    def update_hyperparams(self, epoch):
        """
        Function used to update learning rate and entropy coefficiency to reduce steadily this two params.
        Every 20 epoch are reduced by 10%
        """
        if epoch % 20==0:
            self.learningRate-=self.learningRate*0.1
            self.entropyCoeffiecient-=self.entropyCoeffiecient*0.1
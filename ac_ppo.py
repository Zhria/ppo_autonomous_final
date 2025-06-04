from rete import Actor,Critic
import tensorflow as tf

class ActorCriticPPO():
    def __init__(self,env,optimizer,weigthsActor=None, weigthsCritic=None):
        self.dim_in=env.observation_space.shape
        self.dim_out=env.action_space.n
        self.weightsActorPath=weigthsActor
        self.weightsCriticPath=weigthsCritic        
        self.modelLoaded=False
        try:
            if self.weightsActorPath is not None:
                self.actor=tf.keras.models.load_model(self.weightsActorPath)
                self.modelLoaded=True
            if self.weightsCriticPath is not None:
                self.critic=tf.keras.models.load_model(self.weightsCriticPath)
                self.modelLoaded=self.modelLoaded and True
            print("ACTOR CRITIC LOADED")
        except Exception as e:
            print("--------- EXCEPTION :")
            print(e)
            print("Errore nel caricamento dei pesi")
            self.modelLoaded=False

        if self.modelLoaded==False:
            self.actor=Actor(name="actor_ppo")
            self.critic=Critic(name="critic_ppo")
            self.critic.compile(optimizer=optimizer)
            self.actor.compile(optimizer=optimizer)        

        self.actor.summary()
        self.critic.summary()
        
    def get_trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables
    
    def forward(self,batch_obs):
        policy=self.actor(batch_obs)
        value=self.critic(batch_obs)
        try:
                tf.debugging.check_numerics(policy, "POLICY has nan or inf values")
        except:
                print("POLICY HAS NAN OR INF VALUES")
                print(policy)
                
        try:
                tf.debugging.check_numerics(value, "VALUE has nan or inf values")
        except:
                print("VALUE HAS NAN OR INF VALUES")
                print(value)
                
                    
        return policy,value
    

    def save(self):
        print("pathActor: "+ self.weightsActorPath)
        print("pathCritic: "+ self.weightsCriticPath)
        #self.actor.save_weights(self.weightsActorPath)
        #self.critic.save_weights(self.weightsCriticPath)
        self.actor.save(filepath=self.weightsActorPath)
        self.critic.save(filepath=self.weightsCriticPath)
        return